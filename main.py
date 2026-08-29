import asyncio
import json
import logging
import os
import random
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import aiohttp
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ContentType, ParseMode
from aiogram.filters import BaseFilter, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Message,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from groq import AsyncGroq

# --------------------------------------------------------------------------- #
# КОНФИГУРАЦИЯ / ОКРУЖЕНИЕ
# --------------------------------------------------------------------------- #

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INSTA_ACCOUNT_ID = os.getenv("INSTA_ACCOUNT_ID")
INSTA_ACCESS_TOKEN = os.getenv("INSTA_ACCESS_TOKEN")

if not BOT_TOKEN or not GROQ_API_KEY:
    raise RuntimeError("Переменные окружения BOT_TOKEN и GROQ_API_KEY обязательны")

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)
CONFIG_PATH = BASE_DIR / "config.json"
ASSETS_DIR = BASE_DIR / "assets"
SUBTITLE_FONT_NAME = "DejaVu Sans"
# собственные синтезированные (не сэмплированные из чужой музыки) фоновые подложки —
# без риска авторских прав, т.к. сгенерированы напрямую через аудиофильтры ffmpeg
BG_MUSIC_TRACKS = [ASSETS_DIR / "bg_music_upbeat.mp3", ASSETS_DIR / "bg_music_chill.mp3"]
BG_MUSIC_VOLUME = 0.16

# статичный ffmpeg-бинарник из pip-пакета — не зависит от apt/Aptfile на хостинге
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

GROQ_LLM_MODEL = "openai/gpt-oss-120b"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"
GRAPH_API_VERSION = "v20.0"
GRAPH_API_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}"
CATBOX_API_URL = "https://catbox.moe/user/api.php"
FALLBACK_UPLOAD_URL = "https://0x0.st"
POLLINATIONS_URL = "https://image.pollinations.ai/prompt"
# многие бесплатные файлхостинги режут запросы без "браузерного" User-Agent как ботов
UPLOAD_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
# анонимный доступ к Pollinations ограничен 1 запросом в 15 сек — берём небольшой запас
POLLINATIONS_MIN_INTERVAL = 16.0
POLLINATIONS_MAX_RETRIES = 5


class _RateLimiter:
    """Гарантирует минимальный интервал между стартами запросов к общему ресурсу,
    даже если несколько корутин пытаются обратиться к нему одновременно."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_call = time.monotonic()


pollinations_limiter = _RateLimiter(POLLINATIONS_MIN_INTERVAL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("content-zavod")

groq_client = AsyncGroq(api_key=GROQ_API_KEY)
scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
router = Router()

DEFAULT_CONFIG = {
    "admin_id": None,
    "niche": "Онлайн-обучение и личностный рост",
    "posts_per_day": 2,
    "running": False,
}

_config_lock = asyncio.Lock()


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return {**DEFAULT_CONFIG, **data}
        except (json.JSONDecodeError, OSError):
            logger.exception("Не удалось прочитать config.json, использую значения по умолчанию")
    return dict(DEFAULT_CONFIG)


def _save_config_sync(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


CONFIG = _load_config()


async def save_config() -> None:
    async with _config_lock:
        await asyncio.to_thread(_save_config_sync, CONFIG)


# черновики постов/роликов, ожидающие решения админа (публикация или отмена)
PENDING: dict[str, dict] = {}


def register_pending(kind: str, **data) -> str:
    token = uuid.uuid4().hex[:10]
    PENDING[token] = {"kind": kind, "created": time.time(), **data}
    return token


def pop_pending(token: str) -> Optional[dict]:
    return PENDING.pop(token, None)


def cleanup_pending_files(item: dict) -> None:
    paths = list(item.get("image_paths") or [])
    video_path = item.get("video_path")
    if video_path:
        paths.append(video_path)
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning("Не удалось удалить файл %s", path)


async def cleanup_stale_pending() -> None:
    now = time.time()
    stale = [t for t, item in PENDING.items() if now - item.get("created", now) > 24 * 3600]
    for token in stale:
        item = PENDING.pop(token, None)
        if item:
            cleanup_pending_files(item)
    if stale:
        logger.info("Очищено %d устаревших черновиков", len(stale))


# --------------------------------------------------------------------------- #
# ФИЛЬТР АДМИНА / КЛАВИАТУРЫ / FSM
# --------------------------------------------------------------------------- #


class AdminFilter(BaseFilter):
    async def __call__(self, event) -> bool:
        admin_id = CONFIG.get("admin_id")
        user = event.from_user
        return admin_id is not None and user is not None and user.id == admin_id


class NicheStates(StatesGroup):
    waiting_niche = State()


def main_menu_kb() -> InlineKeyboardMarkup:
    toggle_text = "⏹ Стоп" if CONFIG["running"] else "▶️ Старт"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=toggle_text, callback_data="toggle_scheduler")],
            [
                InlineKeyboardButton(text="🎯 Ниша", callback_data="set_niche"),
                InlineKeyboardButton(text="⏰ Постов в день", callback_data="set_frequency"),
            ],
            [InlineKeyboardButton(text="⚡️ Сделать пост сейчас", callback_data="post_now")],
            [InlineKeyboardButton(text="🎬 AI-слайдшоу сейчас", callback_data="slideshow_now")],
        ]
    )


def status_text() -> str:
    state = "🟢 запущен" if CONFIG["running"] else "🔴 остановлен"
    insta_line = (
        "📸 Instagram: подключён"
        if instagram_configured()
        else "📸 Instagram: 🧪 тестовый режим (нет ключей — публикация будет симулирована)"
    )
    return (
        "🏭 <b>Контент-завод для Instagram</b>\n\n"
        f"Статус автопостинга: {state}\n"
        f"{insta_line}\n"
        f"Ниша: <i>{CONFIG['niche']}</i>\n"
        f"Постов в день: <b>{CONFIG['posts_per_day']}</b>\n\n"
        "Пришлите видео (1–10 минут) — сделаю из него вирусный Reels с субтитрами.\n"
        "Кнопки ниже управляют автопостингом."
    )


# --------------------------------------------------------------------------- #
# ОБЩИЕ ХЕЛПЕРЫ
# --------------------------------------------------------------------------- #


def extract_json(raw: str) -> dict:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Не удалось распарсить JSON из ответа модели: {raw[:300]}")


async def run_ffmpeg(cmd: list[str], timeout: int = 600) -> None:
    logger.info("FFmpeg: %s", " ".join(cmd))
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        raise RuntimeError("FFmpeg превысил лимит времени выполнения")
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg завершился с ошибкой: {stderr.decode(errors='ignore')[-2000:]}")


# ffmpeg конвертирует "голый" .srt в ASS через собственный маленький внутренний PlayRes и потом
# масштабирует его до реального кадра — из-за этого force_style с пиксельными MarginV/FontSize
# рендерился в 6+ раз не в том месте, чем ожидалось (проверено эмпирически: MarginV=90 давал
# отступ ~587px, а не 90). Пишем полноценный .ass с явным PlayResX/PlayResY = кадру видео —
# тогда все размеры в стиле буквально совпадают с реальными пикселями.
ASS_PLAY_RES = (1080, 1920)
ASS_FONT_SIZE = 66
ASS_OUTLINE = 7
ASS_MARGIN_V = 300  # с запасом от нижнего UI Instagram (кэпшн/музыка/кнопки)
ASS_MARGIN_LR = 70

# WrapStyle=0 (умный перенос) обязателен: длинные русские слова на FontSize=66 Bold легко не
# влезают в MarginL/MarginR даже при 2-3 словах в чанке — без переноса libass не оборачивает
# строку, а просто рисует её за пределами кадра ("субтитры убегают" за края).
ASS_HEADER_TEMPLATE = """[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,{outline},0,2,{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def build_ass(
    words: list[dict],
    start: float,
    end: float,
    max_words_per_line: int = 3,
    pause_gap: float = 0.35,
) -> str:
    # короткие 2-3-словные "биты", разрывающиеся по естественным паузам речи — читаются как
    # динамичные вирусные субтитры (слово-в-слово под ритм), а не сплошной подстрочник
    relevant = [w for w in words if w["end"] > start and w["start"] < end]
    events: list[str] = []
    chunk: list[dict] = []

    def flush(buf: list[dict]) -> None:
        if not buf:
            return
        s = max(0.0, buf[0]["start"] - start)
        e = max(s + 0.2, buf[-1]["end"] - start)
        text = " ".join(w["word"].strip() for w in buf).strip().upper()
        events.append(f"Dialogue: 0,{format_ass_time(s)},{format_ass_time(e)},Default,,0,0,0,,{text}")

    for w in relevant:
        if chunk and w["start"] - chunk[-1]["end"] >= pause_gap:
            flush(chunk)
            chunk = []
        chunk.append(w)
        duration = chunk[-1]["end"] - chunk[0]["start"]
        if len(chunk) >= max_words_per_line or duration >= 1.6:
            flush(chunk)
            chunk = []
    flush(chunk)

    header = ASS_HEADER_TEMPLATE.format(
        play_res_x=ASS_PLAY_RES[0],
        play_res_y=ASS_PLAY_RES[1],
        font=SUBTITLE_FONT_NAME,
        font_size=ASS_FONT_SIZE,
        outline=ASS_OUTLINE,
        margin_lr=ASS_MARGIN_LR,
        margin_v=ASS_MARGIN_V,
    )
    return header + "\n".join(events)


def fallback_segments_from_words(words: list[dict], chunk_seconds: float = 8.0) -> list[dict]:
    segments: list[dict] = []
    chunk: list[dict] = []
    chunk_start: Optional[float] = None
    for w in words:
        if chunk_start is None:
            chunk_start = w["start"]
        chunk.append(w)
        if w["end"] - chunk_start >= chunk_seconds:
            segments.append(
                {"start": chunk_start, "end": w["end"], "text": " ".join(x["word"] for x in chunk)}
            )
            chunk = []
            chunk_start = None
    if chunk:
        segments.append(
            {"start": chunk_start, "end": chunk[-1]["end"], "text": " ".join(x["word"] for x in chunk)}
        )
    return segments


def escape_ffmpeg_filter_path(path: str) -> str:
    path = path.replace("\\", "\\\\")
    path = path.replace(":", "\\:")
    path = path.replace("'", "\\'")
    return path


# --------------------------------------------------------------------------- #
# GROQ: ТЕКСТ / ИЗОБРАЖЕНИЕ / ТРАНСКРИБАЦИЯ
# --------------------------------------------------------------------------- #


SLIDE_COUNT_MIN = 4
SLIDE_COUNT_MAX = 6


async def generate_post_content(niche: str) -> dict:
    system_prompt = (
        "Ты — тандем из топового SMM-копирайтера и арт-директора, который ставит техзадания "
        "фотографу для съёмки Instagram-карусели уровня рекламной кампании (не студенческий "
        "стоковый банк). Всегда отвечай строго валидным JSON без пояснений."
    )
    user_prompt = f"""
Ниша/продукт: {niche}

Собери карусель из Instagram-поста на 5 слайдов строго по формуле Hook-Story-Offer:
- "caption": общий продающий текст поста на русском (Hook в первой строке, короткая история/боль
  клиента, чёткий offer с призывом к действию и лёгким дедлайном/бонусом), 5-8 хэштегов, уместные
  эмодзи, до 900 символов.
- "slides": массив РОВНО из 5 объектов, по одному на каждый слайд карусели, в таком порядке:
  1) Hook-слайд — самый цепляющий, останавливающий скролл в ленте.
  2-4) Story-слайды — раскрывают боль клиента/процесс/пользу продукта, каждый со своим ракурсом
  и сценой, без повторов.
  5) Offer-слайд — призыв к действию.

Для каждого слайда пиши "image_prompt" (на английском, для генеративной модели FLUX) как
техзадание фотографу для дорогой кинематографичной ПРЕДМЕТНОЙ/символической съёмки — в стиле
премиального научно-популярного документального фильма или рекламы фармацевтического бренда
(тёмный фон, один драматичный источник света, глубокий боке) — а НЕ повседневное фото человека
с ноутбуком. ОБЯЗАТЕЛЬНО включи все пункты подряд, в одном предложении-рецепте:
1. Один конкретный ПРЕДМЕТ или символ, который визуально олицетворяет именно факт/мысль ЭТОГО
   слайда (не абстракция) — например: таблетка между пальцами в медицинской перчатке, светящаяся
   объёмная модель ДНК на столе, лабораторная мышь под тёплым светом, капли на пробирке, песочные
   часы, старинные карманные часы рядом с молекулой, микроскоп с подсветкой. Каждый слайд —
   разный предмет, без повторов.
2. Драматичное студийное освещение: один тёплый источник света сбоку или сверху (warm spotlight
   from above, single soft key light), густая тень, глубокий чёрный/тёмно-бордовый фон не в
   фокусе.
3. Операторские детали: "macro close-up shot, extremely shallow depth of field, cinematic
   lighting, shot on 85mm lens, shot like a Netflix science documentary title card" — конкретно,
   разное для каждого слайда.
4. Композиция: "extreme close-up", "shallow focus with foreground blur", "top-down macro shot" —
   меняй ракурс от слайда к слайду.
Обязательно заверши каждый image_prompt фразой: "photorealistic, cinematic, high detail, moody
dark background, no text, no watermark, no logo, no deformed hands".
Плохой пример (так не пиши): "A person working on a laptop, natural light, professional photo".
Хороший пример: "A single white pill held between two fingers in a blue medical glove, extreme
macro close-up, dark blurred laboratory background with warm amber rim light from one side,
shallow depth of field, cinematic lighting, shot on 85mm lens like a pharmaceutical commercial,
photorealistic, cinematic, high detail, moody dark background, no text, no watermark, no logo,
no deformed hands".

Также для каждого слайда пиши "hook_text": содержательный факт/тезис на русском (1-2 короткие
фразы, до 160 символов) — как подпись-инфографика поверх фото в стиле научно-популярных
каруселей: конкретная цифра/факт/вывод, а не общий лозунг. Внутри hook_text выдели САМОЕ
важное слово или фразу (цифру, ключевой термин) двойными звёздочками **вот так** — это будет
подсвечено оранжевым цветом на картинке, остальной текст останется белым. Пример:
"В опытах на мышах продолжительность жизни **выросла на 25%**, а возрастные болезни появлялись
**значительно позже**".

Верни строго JSON:
{{"caption": "<текст поста>", "slides": [{{"image_prompt": "...", "hook_text": "..."}}, ...]}}
"""
    completion = await groq_client.chat.completions.create(
        model=GROQ_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
        max_tokens=5000,
        response_format={"type": "json_object"},
    )
    data = extract_json(completion.choices[0].message.content)
    slides = data.get("slides")
    if "caption" not in data or not isinstance(slides, list):
        raise ValueError(f"Некорректный ответ LLM: {data}")
    if len(slides) > SLIDE_COUNT_MAX:
        slides = slides[:SLIDE_COUNT_MAX]
    if len(slides) < SLIDE_COUNT_MIN:
        raise ValueError(f"LLM вернул слишком мало слайдов карусели ({len(slides)}): {data}")
    for slide in slides:
        if "image_prompt" not in slide or "hook_text" not in slide:
            raise ValueError(f"Некорректный слайд карусели в ответе LLM: {slide}")
    data["slides"] = slides
    return data


async def fetch_flux_image(prompt: str) -> Image.Image:
    seed = random.randint(1, 2_000_000_000)
    encoded_prompt = quote(f"{prompt}, 8k, sharp focus, no text, no watermark, no logo")
    url = f"{POLLINATIONS_URL}/{encoded_prompt}?width=1080&height=1080&nologo=true&model=flux&seed={seed}"
    raw_path = TEMP_DIR / f"raw_{uuid.uuid4().hex}.jpg"

    for attempt in range(POLLINATIONS_MAX_RETRIES):
        await pollinations_limiter.wait()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=180)) as resp:
                    resp.raise_for_status()
                    with open(raw_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(65536):
                            f.write(chunk)
            if raw_path.stat().st_size < 1024:
                raise RuntimeError("Pollinations вернул пустое изображение")
            break
        except (aiohttp.ClientError, RuntimeError) as exc:
            raw_path.unlink(missing_ok=True)
            if attempt == POLLINATIONS_MAX_RETRIES - 1:
                raise
            backoff = 2**attempt
            logger.warning("Pollinations запрос не удался (%s), повтор через %sс", exc, backoff)
            await asyncio.sleep(backoff)

    def _load_and_fit() -> Image.Image:
        with Image.open(raw_path) as img:
            img = img.convert("RGB")
            return ImageOps.fit(img, (1080, 1080), method=Image.Resampling.LANCZOS)

    image = await asyncio.to_thread(_load_and_fit)
    raw_path.unlink(missing_ok=True)
    return image


def compose_vertical_frame(image: Image.Image, size: tuple[int, int] = (1080, 1920)) -> Image.Image:
    # квадратное AI-фото -> кадр 9:16: чёткая версия по центру вписана целиком (ничего не
    # обрезано), поля сверху/снизу — размытая увеличенная копия того же фото. Та же идея, что и
    # у render_reel для видео с исходником, только тут делается через Pillow на статичной картинке.
    target_w, target_h = size
    background = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
    background = background.filter(ImageFilter.GaussianBlur(40))
    canvas = background.convert("RGB")

    foreground = image.copy()
    foreground.thumbnail(size, Image.Resampling.LANCZOS)
    paste_x = (target_w - foreground.width) // 2
    paste_y = (target_h - foreground.height) // 2
    canvas.paste(foreground, (paste_x, paste_y))
    return canvas


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        bbox = font.getbbox(trial)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_outlined_text(
    draw: "ImageDraw.ImageDraw", xy: tuple[float, float], text: str, font: ImageFont.FreeTypeFont
) -> None:
    x, y = xy
    for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3), (-2, -2), (2, 2), (-2, 2), (2, -2)):
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0, 255))
    draw.text((x, y), text, font=font, fill=(255, 221, 0, 255))


HOOK_COLOR_NORMAL = (255, 255, 255, 255)
HOOK_COLOR_EMPHASIS = (255, 149, 0, 255)  # тёплый оранжевый — подсветка ключевых слов


def _parse_emphasis_words(text: str) -> list[tuple[str, bool]]:
    # разбирает "обычный **выделенный** текст" на список (слово, выделено_ли)
    tokens: list[tuple[str, bool]] = []
    for part in re.split(r"(\*\*.*?\*\*)", text):
        if not part:
            continue
        emphasized = part.startswith("**") and part.endswith("**") and len(part) > 4
        clean = part[2:-2] if emphasized else part
        for word in clean.split():
            tokens.append((word, emphasized))

    # знак препинания сразу после закрывающего "**" превращается в отдельный "словотокен" —
    # без этого перед запятой/точкой рисуется лишний пробел
    merged: list[tuple[str, bool]] = []
    for word, emph in tokens:
        if merged and re.fullmatch(r"[,.!?;:…]+", word):
            prev_word, prev_emph = merged[-1]
            merged[-1] = (prev_word + word, prev_emph)
        else:
            merged.append((word, emph))
    return merged


def _wrap_emphasis_words(
    words: list[tuple[str, bool]], font: ImageFont.FreeTypeFont, max_width: int
) -> list[list[tuple[str, bool]]]:
    lines: list[list[tuple[str, bool]]] = []
    current: list[tuple[str, bool]] = []
    for word, emph in words:
        trial = current + [(word, emph)]
        trial_text = " ".join(w for w, _ in trial)
        if font.getlength(trial_text) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = [(word, emph)]
    if current:
        lines.append(current)
    return lines


def _compose_hook_overlay(base: Image.Image, text: str, show_swipe_hint: bool) -> Image.Image:
    # base должен быть RGBA нужного размера — либо реальное фото (карусель), либо полностью
    # прозрачный холст (видео-оверлей поверх Ken Burns-анимации). Возвращает RGBA с текстом.
    width, height = base.size
    margin_x = 64
    font_size = 50
    font = ImageFont.truetype(str(ASSETS_DIR / "DejaVuSans-Bold.ttf"), font_size)

    words = _parse_emphasis_words(text)
    lines = _wrap_emphasis_words(words, font, width - margin_x * 2)

    line_height = font_size + 14
    bottom_pad = 90
    block_height = line_height * len(lines)
    text_top = height - bottom_pad - block_height

    # мягкий градиент снизу (а не сплошная плашка) — читаемо на любом фоне, но не "заклеивает" фото
    gradient_start = max(0, text_top - 130)
    grad = Image.new("L", (1, height), 0)
    for y in range(height):
        if y <= gradient_start:
            grad.putpixel((0, y), 0)
        else:
            frac = (y - gradient_start) / max(1, height - gradient_start)
            grad.putpixel((0, y), int(215 * min(1.0, frac)))
    grad = grad.resize((width, height))
    scrim = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    scrim.putalpha(grad)

    composed = Image.alpha_composite(base, scrim)
    draw = ImageDraw.Draw(composed)

    space_width = font.getlength(" ")
    y = text_top
    for line in lines:
        x = float(margin_x)
        for word, emph in line:
            color = HOOK_COLOR_EMPHASIS if emph else HOOK_COLOR_NORMAL
            draw.text((x + 2, y + 2), word, font=font, fill=(0, 0, 0, 150))
            draw.text((x, y), word, font=font, fill=color)
            x += font.getlength(word) + space_width
        y += line_height

    if show_swipe_hint:
        hint_font = ImageFont.truetype(str(ASSETS_DIR / "DejaVuSans-Bold.ttf"), 26)
        hint_text = "ЛИСТАЙ →"
        hint_y = height - 54
        draw.text((margin_x + 2, hint_y + 2), hint_text, font=hint_font, fill=(0, 0, 0, 150))
        draw.text((margin_x, hint_y), hint_text, font=hint_font, fill=(255, 255, 255, 220))

    return composed


def draw_hook_text(image: Image.Image, text: str, show_swipe_hint: bool = True) -> Image.Image:
    composed = _compose_hook_overlay(image.convert("RGBA"), text, show_swipe_hint)
    return composed.convert("RGB")


def render_slide_overlay_png(text: str, size: tuple[int, int] = (1080, 1920)) -> Path:
    # прозрачный PNG с той же текстовой подачей, что и на карусели (для оверлея на видео-слайдшоу)
    base = Image.new("RGBA", size, (0, 0, 0, 0))
    composed = _compose_hook_overlay(base, text, show_swipe_hint=False)
    out_path = TEMP_DIR / f"slidetext_{uuid.uuid4().hex}.png"
    composed.save(out_path, format="PNG")
    return out_path


def render_hook_overlay_png(hook_text: str, size: tuple[int, int] = (1080, 1920)) -> Path:
    width, height = size
    font_size = 84
    font = ImageFont.truetype(str(ASSETS_DIR / "DejaVuSans-Bold.ttf"), font_size)
    lines = _wrap_text(hook_text, font, width - 140)

    line_height = font_size + 18
    block_height = line_height * len(lines) + 70
    top_offset = 160  # верхняя треть кадра — не перекрывает субтитры снизу

    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(canvas).rectangle(
        [0, top_offset, width, top_offset + block_height], fill=(0, 0, 0, 165)
    )
    draw = ImageDraw.Draw(canvas)

    y = top_offset + 35
    for line in lines:
        bbox = font.getbbox(line)
        x = (width - (bbox[2] - bbox[0])) / 2
        _draw_outlined_text(draw, (x, y), line, font)
        y += line_height

    out_path = TEMP_DIR / f"hook_{uuid.uuid4().hex}.png"
    canvas.save(out_path, format="PNG")
    return out_path


async def generate_slide_image(image_prompt: str, hook_text: str, show_swipe_hint: bool) -> Path:
    image = await fetch_flux_image(image_prompt)
    final_path = TEMP_DIR / f"slide_{uuid.uuid4().hex}.jpg"

    def _draw_and_save() -> None:
        framed = draw_hook_text(image, hook_text, show_swipe_hint=show_swipe_hint)
        framed.save(final_path, format="JPEG", quality=92)

    await asyncio.to_thread(_draw_and_save)
    return final_path


async def generate_carousel_slides(slides: list[dict]) -> list[Path]:
    last_index = len(slides) - 1
    tasks = [
        asyncio.create_task(
            generate_slide_image(slide["image_prompt"], slide["hook_text"], i != last_index)
        )
        for i, slide in enumerate(slides)
    ]
    try:
        return list(await asyncio.gather(*tasks))
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            if not task.cancelled() and task.exception() is None:
                path = task.result()
                if path.exists():
                    path.unlink(missing_ok=True)
        raise


async def transcribe_audio(audio_path: Path) -> dict:
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    transcript = await groq_client.audio.transcriptions.create(
        file=(audio_path.name, audio_bytes),
        model=GROQ_WHISPER_MODEL,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
    return transcript.model_dump() if hasattr(transcript, "model_dump") else dict(transcript)


MAX_CLIPS_PER_VIDEO = 3
MIN_CLIP_SECONDS = 20
MAX_CLIP_SECONDS = 50


async def pick_viral_segments(niche: str, segments: list[dict], duration: float) -> list[dict]:
    lines = [f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text'].strip()}" for seg in segments]
    transcript_block = "\n".join(lines)
    max_clips = min(MAX_CLIPS_PER_VIDEO, max(1, int(duration // MIN_CLIP_SECONDS)))
    system_prompt = (
        "Ты — вирусный видеоредактор и SMM-стратег, который находит самые цепляющие "
        "20-50-секундные фрагменты в длинных видео для Reels. Отвечай строго валидным JSON."
    )
    user_prompt = f"""
Ниша: {niche}
Общая длительность видео: {duration:.1f} сек.

Транскрипт с таймкодами (секунды):
{transcript_block}

Выбери до {max_clips} САМЫХ вирусных, экспертных или эмоциональных НЕПЕРЕСЕКАЮЩИХСЯ фрагментов
длиной от {MIN_CLIP_SECONDS} до {MAX_CLIP_SECONDS} секунд каждый (можно меньше {max_clips}, если
в видео реально нет столько ярких моментов — не тяни за уши). Тайминги start/end должны попадать
в пределы видео (0..{duration:.1f}), совпадать с границами реплик из транскрипта и НЕ пересекаться
между собой.

Для каждого фрагмента напиши:
- "caption": продающий кэпшн на русском с сильным хуком в первой строке и призывом к действию,
  до 500 символов (кратко!), с эмодзи и 3-5 хэштегами (это текст ПОД видео в Instagram).
- "hook_text": короткая цепляющая фраза на русском (3-7 слов, БЕЗ хэштегов и эмодзи) — крупная
  надпись поверх первых секунд самого видео, чтобы остановить скролл. Не дублируй дословно первую
  строку caption, сформулируй ударнее и короче.

Верни строго JSON: {{"clips": [{{"start": <секунды>, "end": <секунды>, "caption": "<текст>", "hook_text": "<текст>"}}, ...]}}
Отсортируй clips от самого вирусного к наименее вирусному.
"""
    completion = await groq_client.chat.completions.create(
        model=GROQ_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=6000,
        response_format={"type": "json_object"},
    )
    data = extract_json(completion.choices[0].message.content)
    clips = data.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError(f"Некорректный ответ LLM: {data}")

    picked: list[dict] = []
    occupied: list[tuple[float, float]] = []
    for clip in clips[:max_clips]:
        if "hook_text" not in clip or "caption" not in clip:
            continue
        start = max(0.0, float(clip["start"]))
        end = min(duration, float(clip["end"]))
        if end - start < 5:
            continue
        if end - start > MAX_CLIP_SECONDS:
            end = start + MAX_CLIP_SECONDS
        if any(start < o_end and end > o_start for o_start, o_end in occupied):
            continue
        occupied.append((start, end))
        picked.append(
            {"start": start, "end": end, "caption": clip["caption"], "hook_text": clip["hook_text"]}
        )

    if not picked:
        raise ValueError(f"LLM не вернул ни одного валидного фрагмента: {data}")
    return picked


HOOK_OVERLAY_SECONDS = 2.5


async def render_reel(source: Path, ass_path: Path, start: float, end: float, hook_text: str) -> Path:
    output_path = TEMP_DIR / f"reel_{uuid.uuid4().hex}.mp4"
    subtitles_arg = escape_ffmpeg_filter_path(str(ass_path))
    fontsdir_arg = escape_ffmpeg_filter_path(str(ASSETS_DIR))
    # fontsdir указывает на шрифт, вложенный прямо в репозиторий (assets/), чтобы не зависеть
    # от системных шрифтов хостинга. Сам стиль (шрифт/цвет/отступы) задан внутри .ass-файла
    # (build_ass), а не через force_style — см. комментарий там про баг с масштабированием.
    # Раньше кадр жёстко кропался под 9:16 — для видео с непортретными пропорциями (обычная
    # горизонтальная съёмка) это вырезало значительную часть кадра и "приближало"/обрезало
    # человека. Теперь вместо кропа видео вписывается ЦЕЛИКОМ (ничего не теряется), а пустые
    # поля сверху/снизу или по бокам заполняются размытой увеличенной копией того же кадра —
    # так делают CapCut/Kapwing и подобные редакторы, вместо голых чёрных полос.
    video_chain = (
        "[0:v]split=2[bg][fg];"
        "[bg]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,gblur=sigma=25[bgblur];"
        "[fg]scale=1080:1920:force_original_aspect_ratio=decrease[fgscaled];"
        "[bgblur][fgscaled]overlay=(W-w)/2:(H-h)/2,"
        f"subtitles='{subtitles_arg}':fontsdir='{fontsdir_arg}'[base]"
    )

    hook_png = await asyncio.to_thread(render_hook_overlay_png, hook_text)
    bg_music_path = random.choice(BG_MUSIC_TRACKS)
    try:
        filter_complex = (
            f"{video_chain};"
            f"[base][1:v]overlay=0:0:enable='between(t,0,{HOOK_OVERLAY_SECONDS})'[outv];"
            f"[2:a]volume={BG_MUSIC_VOLUME}[bgm];"
            "[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[outa]"
        )
        await run_ffmpeg(
            [
                FFMPEG_BIN, "-y",
                "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                "-i", str(source),
                "-i", str(hook_png),
                "-stream_loop", "-1", "-i", str(bg_music_path),
                "-filter_complex", filter_complex,
                "-map", "[outv]", "-map", "[outa]",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                "-movflags", "+faststart",
                str(output_path),
            ],
            timeout=900,
        )
    finally:
        hook_png.unlink(missing_ok=True)
    return output_path


# --------------------------------------------------------------------------- #
# CATBOX / INSTAGRAM GRAPH API
# --------------------------------------------------------------------------- #


async def upload_to_catbox(file_path: str) -> str:
    filename = os.path.basename(file_path)
    headers = {"User-Agent": UPLOAD_USER_AGENT}
    async with aiohttp.ClientSession(headers=headers) as session:
        with open(file_path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("reqtype", "fileupload")
            form.add_field("fileToUpload", f, filename=filename)
            async with session.post(
                CATBOX_API_URL, data=form, timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                text = (await resp.text()).strip()
                if resp.status != 200 or not text.startswith("http"):
                    raise RuntimeError(f"Catbox не вернул ссылку: {text[:300]}")
                return text


async def upload_to_fallback_host(file_path: str) -> str:
    headers = {"User-Agent": UPLOAD_USER_AGENT}
    async with aiohttp.ClientSession(headers=headers) as session:
        with open(file_path, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("file", f, filename=os.path.basename(file_path))
            async with session.post(
                FALLBACK_UPLOAD_URL, data=form, timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                text = (await resp.text()).strip()
                if resp.status != 200 or not text.startswith("http"):
                    raise RuntimeError(f"Резервный хостинг не вернул ссылку: {text[:300]}")
                return text


async def upload_media(file_path: str) -> str:
    try:
        return await upload_to_catbox(file_path)
    except Exception:
        logger.exception("Catbox отклонил загрузку, пробую резервный хостинг")
        return await upload_to_fallback_host(file_path)


def instagram_configured() -> bool:
    return bool(INSTA_ACCOUNT_ID and INSTA_ACCESS_TOKEN)


def require_instagram_credentials() -> None:
    if not instagram_configured():
        raise RuntimeError(
            "Не заданы INSTA_ACCOUNT_ID / INSTA_ACCESS_TOKEN — публикация в Instagram невозможна"
        )


async def publish_photo_to_instagram(image_url: str, caption: str) -> str:
    require_instagram_credentials()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{GRAPH_API_URL}/{INSTA_ACCOUNT_ID}/media",
            data={"image_url": image_url, "caption": caption, "access_token": INSTA_ACCESS_TOKEN},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            payload = await resp.json()
            if "id" not in payload:
                raise RuntimeError(f"Instagram media error: {payload}")
            creation_id = payload["id"]

        async with session.post(
            f"{GRAPH_API_URL}/{INSTA_ACCOUNT_ID}/media_publish",
            data={"creation_id": creation_id, "access_token": INSTA_ACCESS_TOKEN},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            payload = await resp.json()
            if "id" not in payload:
                raise RuntimeError(f"Instagram publish error: {payload}")
            return payload["id"]


async def publish_carousel_to_instagram(image_urls: list[str], caption: str) -> str:
    require_instagram_credentials()
    async with aiohttp.ClientSession() as session:
        child_ids = []
        for image_url in image_urls:
            async with session.post(
                f"{GRAPH_API_URL}/{INSTA_ACCOUNT_ID}/media",
                data={
                    "image_url": image_url,
                    "is_carousel_item": "true",
                    "access_token": INSTA_ACCESS_TOKEN,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                payload = await resp.json()
                if "id" not in payload:
                    raise RuntimeError(f"Instagram carousel child error: {payload}")
                child_ids.append(payload["id"])

        async with session.post(
            f"{GRAPH_API_URL}/{INSTA_ACCOUNT_ID}/media",
            data={
                "media_type": "CAROUSEL",
                "children": ",".join(child_ids),
                "caption": caption,
                "access_token": INSTA_ACCESS_TOKEN,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            payload = await resp.json()
            if "id" not in payload:
                raise RuntimeError(f"Instagram carousel container error: {payload}")
            creation_id = payload["id"]

        async with session.post(
            f"{GRAPH_API_URL}/{INSTA_ACCOUNT_ID}/media_publish",
            data={"creation_id": creation_id, "access_token": INSTA_ACCESS_TOKEN},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            payload = await resp.json()
            if "id" not in payload:
                raise RuntimeError(f"Instagram publish (carousel) error: {payload}")
            return payload["id"]


async def publish_reel_to_instagram(video_url: str, caption: str) -> str:
    require_instagram_credentials()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{GRAPH_API_URL}/{INSTA_ACCOUNT_ID}/media",
            data={
                "media_type": "REELS",
                "video_url": video_url,
                "caption": caption,
                "share_to_feed": "true",
                "access_token": INSTA_ACCESS_TOKEN,
            },
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            payload = await resp.json()
            if "id" not in payload:
                raise RuntimeError(f"Instagram media (reels) error: {payload}")
            creation_id = payload["id"]

        await asyncio.sleep(15)

        status = "IN_PROGRESS"
        payload = {}
        for _ in range(20):
            async with session.get(
                f"{GRAPH_API_URL}/{creation_id}",
                params={"fields": "status_code", "access_token": INSTA_ACCESS_TOKEN},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                payload = await resp.json()
                status = payload.get("status_code", "IN_PROGRESS")
            if status == "FINISHED":
                break
            if status == "ERROR":
                raise RuntimeError(f"Instagram не смог обработать видео: {payload}")
            await asyncio.sleep(5)
        else:
            raise RuntimeError("Instagram слишком долго обрабатывает видео (timeout)")

        async with session.post(
            f"{GRAPH_API_URL}/{INSTA_ACCOUNT_ID}/media_publish",
            data={"creation_id": creation_id, "access_token": INSTA_ACCESS_TOKEN},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            payload = await resp.json()
            if "id" not in payload:
                raise RuntimeError(f"Instagram publish (reels) error: {payload}")
            return payload["id"]


# --------------------------------------------------------------------------- #
# РЕЖИМ 1: АВТО-ПОСТЫ
# --------------------------------------------------------------------------- #


async def generate_and_send_post(bot: Bot) -> None:
    admin_id = CONFIG.get("admin_id")
    if not admin_id:
        return

    try:
        post = await generate_post_content(CONFIG["niche"])
    except Exception:
        logger.exception("Ошибка генерации текста поста")
        await bot.send_message(admin_id, "⚠️ Не удалось сгенерировать пост. Попробую в следующий раз.")
        return

    try:
        slide_paths = await generate_carousel_slides(post["slides"])
    except Exception:
        logger.exception("Ошибка генерации изображений карусели")
        await bot.send_message(admin_id, "⚠️ Не удалось сгенерировать фото для карусели.")
        return

    token = register_pending(
        "carousel", image_paths=[str(p) for p in slide_paths], caption=post["caption"]
    )
    media = [InputMediaPhoto(media=FSInputFile(p)) for p in slide_paths]
    media[0].caption = post["caption"][:1024]
    await bot.send_media_group(admin_id, media=media)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🚀 Опубликовать в Instagram", callback_data=f"publish:{token}"),
                InlineKeyboardButton(text="🔄 Отмена / Переделать", callback_data=f"cancel:{token}"),
            ]
        ]
    )
    await bot.send_message(admin_id, f"👆 Карусель из {len(slide_paths)} фото готова.", reply_markup=kb)


def schedule_autopost(bot: Bot) -> None:
    if scheduler.get_job("autopost"):
        scheduler.remove_job("autopost")
    if CONFIG["running"] and CONFIG["posts_per_day"] > 0:
        interval_seconds = max(1, int(24 * 3600 / CONFIG["posts_per_day"]))
        scheduler.add_job(
            generate_and_send_post,
            trigger=IntervalTrigger(seconds=interval_seconds),
            args=[bot],
            id="autopost",
            replace_existing=True,
            next_run_time=datetime.now() + timedelta(seconds=10),
        )


# --------------------------------------------------------------------------- #
# РЕЖИМ 2: AI-НАРЕЗКА ВИДЕО В REELS
# --------------------------------------------------------------------------- #


async def process_video(bot: Bot, message: Message, file_id: str) -> None:
    status = await message.reply("📥 Скачиваю видео...")

    try:
        tg_file = await bot.get_file(file_id)
    except Exception as exc:
        await status.edit_text(
            "❌ Не удалось скачать видео: "
            f"{exc}\n\nTelegram Bot API ограничивает скачивание файлов до 20 МБ. "
            "Пришлите более короткое или сильнее сжатое видео."
        )
        return

    raw_path = TEMP_DIR / f"src_{uuid.uuid4().hex}.mp4"
    audio_path: Optional[Path] = None
    words: list[dict] = []
    picks: list[dict] = []

    try:
        await bot.download_file(tg_file.file_path, destination=raw_path)

        await status.edit_text("🎙 Извлекаю аудио и транскрибирую через Groq Whisper...")
        audio_path = TEMP_DIR / f"{raw_path.stem}.mp3"
        await run_ffmpeg(
            [
                FFMPEG_BIN, "-y", "-i", str(raw_path),
                "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
                str(audio_path),
            ]
        )

        transcript = await transcribe_audio(audio_path)
        words = transcript.get("words") or []
        segments = transcript.get("segments") or []
        duration = float(transcript.get("duration") or (words[-1]["end"] if words else 0))
        if not words or duration <= 0:
            raise RuntimeError("Groq Whisper не вернул распознанный текст")
        if not segments:
            segments = fallback_segments_from_words(words)

        await status.edit_text("🧠 Ищу самые вирусные моменты и пишу кэпшны...")
        picks = await pick_viral_segments(CONFIG["niche"], segments, duration)
    except Exception as exc:
        logger.exception("Ошибка обработки видео")
        await status.edit_text(f"❌ Ошибка обработки видео: {exc}")
        for path in (raw_path, audio_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        return

    await status.edit_text(
        f"✂️ Нашёл {len(picks)} момент(ов) — нарезаю, кроплю в 9:16, жгу субтитры "
        "и подмешиваю фоновую музыку..."
    )

    sent = 0
    for i, pick in enumerate(picks, start=1):
        ass_path: Optional[Path] = None
        reel_path: Optional[Path] = None
        try:
            ass_path = TEMP_DIR / f"sub_{uuid.uuid4().hex}.ass"
            ass_path.write_text(build_ass(words, pick["start"], pick["end"]), encoding="utf-8")

            reel_path = await render_reel(
                raw_path, ass_path, pick["start"], pick["end"], pick["hook_text"]
            )

            token = register_pending("video", video_path=str(reel_path), caption=pick["caption"])
            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [
                        InlineKeyboardButton(
                            text="🚀 Опубликовать в Instagram", callback_data=f"publish:{token}"
                        ),
                        InlineKeyboardButton(
                            text="🔄 Отмена / Переделать", callback_data=f"cancel:{token}"
                        ),
                    ]
                ]
            )
            await bot.send_video(
                message.chat.id,
                FSInputFile(reel_path),
                caption=f"({i}/{len(picks)}) " + pick["caption"][:1024],
                reply_markup=kb,
                supports_streaming=True,
            )
            sent += 1
        except Exception:
            logger.exception("Ошибка рендера клипа %d/%d", i, len(picks))
            if reel_path and os.path.exists(reel_path):
                os.remove(reel_path)
            await message.reply(f"⚠️ Не удалось собрать клип {i} из {len(picks)} — пропускаю.")
        finally:
            if ass_path and os.path.exists(ass_path):
                try:
                    os.remove(ass_path)
                except OSError:
                    pass

    if sent:
        await status.delete()
    else:
        await status.edit_text("❌ Не удалось собрать ни одного клипа из этого видео.")

    for path in (raw_path, audio_path):
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# РЕЖИМ 3: AI-СЛАЙДШОУ (видео без исходника, целиком из сгенерированных картинок)
# --------------------------------------------------------------------------- #

SLIDESHOW_SLIDE_SECONDS = 5.0
SLIDESHOW_FPS = 25
SLIDESHOW_ZOOM_STEP = 0.0012  # даёт умеренный зум ~1.15x за SLIDESHOW_SLIDE_SECONDS
SLIDESHOW_MUSIC_VOLUME = 0.5  # тут музыка — единственный звук, не нужно уступать место голосу


async def fetch_slideshow_images(slides: list[dict]) -> list[Image.Image]:
    tasks = [asyncio.create_task(fetch_flux_image(slide["image_prompt"])) for slide in slides]
    try:
        return list(await asyncio.gather(*tasks))
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def render_slideshow_clip(
    image: Image.Image, hook_text: str, duration: float, show_swipe_hint: bool
) -> Path:
    frame_path = TEMP_DIR / f"vframe_{uuid.uuid4().hex}.jpg"
    clip_path = TEMP_DIR / f"clip_{uuid.uuid4().hex}.mp4"

    def _compose_frame() -> None:
        compose_vertical_frame(image).save(frame_path, format="JPEG", quality=92)

    await asyncio.to_thread(_compose_frame)
    overlay_path = await asyncio.to_thread(render_slide_overlay_png, hook_text, (1080, 1920))

    try:
        frame_count = max(1, int(duration * SLIDESHOW_FPS))
        zoom_expr = f"min(zoom+{SLIDESHOW_ZOOM_STEP},1.15)"
        # апскейл вдвое перед zoompan сглаживает субпиксельное дрожание при медленном зуме
        filter_complex = (
            "[0:v]scale=2160:3840,"
            f"zoompan=z='{zoom_expr}':d={frame_count}:"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"s=1080x1920:fps={SLIDESHOW_FPS}[zoomed];"
            "[zoomed][1:v]overlay=0:0[outv]"
        )
        await run_ffmpeg(
            [
                FFMPEG_BIN, "-y",
                "-loop", "1", "-i", str(frame_path),
                "-i", str(overlay_path),
                "-filter_complex", filter_complex,
                "-map", "[outv]",
                "-t", f"{duration:.2f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                str(clip_path),
            ],
            timeout=300,
        )
    finally:
        frame_path.unlink(missing_ok=True)
        overlay_path.unlink(missing_ok=True)
    return clip_path


def _escape_concat_path(path: str) -> str:
    return path.replace("'", "'\\''")


async def concat_video_clips(clip_paths: list[Path]) -> Path:
    list_path = TEMP_DIR / f"concat_{uuid.uuid4().hex}.txt"
    merged_path = TEMP_DIR / f"merged_{uuid.uuid4().hex}.mp4"
    list_path.write_text(
        "\n".join(f"file '{_escape_concat_path(str(p))}'" for p in clip_paths) + "\n",
        encoding="utf-8",
    )
    try:
        await run_ffmpeg(
            [
                FFMPEG_BIN, "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-c", "copy",
                str(merged_path),
            ],
            timeout=300,
        )
    finally:
        list_path.unlink(missing_ok=True)
    return merged_path


async def mix_slideshow_music(video_path: Path) -> Path:
    bg_music_path = random.choice(BG_MUSIC_TRACKS)
    output_path = TEMP_DIR / f"slideshow_{uuid.uuid4().hex}.mp4"
    await run_ffmpeg(
        [
            FFMPEG_BIN, "-y",
            "-i", str(video_path),
            "-stream_loop", "-1", "-i", str(bg_music_path),
            "-filter_complex", f"[1:a]volume={SLIDESHOW_MUSIC_VOLUME}[bgm]",
            "-map", "0:v", "-map", "[bgm]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            str(output_path),
        ],
        timeout=300,
    )
    return output_path


async def generate_and_send_slideshow(bot: Bot) -> None:
    admin_id = CONFIG.get("admin_id")
    if not admin_id:
        return

    try:
        post = await generate_post_content(CONFIG["niche"])
    except Exception:
        logger.exception("Ошибка генерации сценария слайд-шоу")
        await bot.send_message(admin_id, "⚠️ Не удалось сгенерировать сценарий слайд-шоу.")
        return

    try:
        images = await fetch_slideshow_images(post["slides"])
    except Exception:
        logger.exception("Ошибка генерации изображений слайд-шоу")
        await bot.send_message(admin_id, "⚠️ Не удалось сгенерировать изображения для слайд-шоу.")
        return

    clip_paths: list[Path] = []
    merged_path: Optional[Path] = None
    final_path: Optional[Path] = None
    try:
        # "ЛИСТАЙ →" тут не нужен ни на одном слайде — это видео, не карусель, листать нечего
        for image, slide in zip(images, post["slides"]):
            clip = await render_slideshow_clip(
                image, slide["hook_text"], SLIDESHOW_SLIDE_SECONDS, False
            )
            clip_paths.append(clip)

        merged_path = await concat_video_clips(clip_paths)
        final_path = await mix_slideshow_music(merged_path)
    except Exception:
        logger.exception("Ошибка сборки слайд-шоу")
        await bot.send_message(admin_id, "⚠️ Не удалось собрать видео-слайдшоу.")
        return
    finally:
        for p in clip_paths:
            p.unlink(missing_ok=True)
        if merged_path:
            merged_path.unlink(missing_ok=True)

    token = register_pending("video", video_path=str(final_path), caption=post["caption"])
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🚀 Опубликовать в Instagram", callback_data=f"publish:{token}"),
                InlineKeyboardButton(text="🔄 Отмена / Переделать", callback_data=f"cancel:{token}"),
            ]
        ]
    )
    await bot.send_video(
        admin_id,
        FSInputFile(final_path),
        caption=post["caption"][:1024],
        reply_markup=kb,
        supports_streaming=True,
    )


# --------------------------------------------------------------------------- #
# ХЕНДЛЕРЫ
# --------------------------------------------------------------------------- #


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    if CONFIG["admin_id"] is None:
        CONFIG["admin_id"] = message.from_user.id
        await save_config()
        await message.answer("👋 Бот активирован. Вы назначены администратором контент-завода.")
    elif CONFIG["admin_id"] != message.from_user.id:
        await message.answer("⛔ Этот бот уже привязан к другому администратору.")
        return
    await message.answer(status_text(), reply_markup=main_menu_kb())


@router.callback_query(AdminFilter(), F.data == "toggle_scheduler")
async def cb_toggle(call: CallbackQuery, bot: Bot) -> None:
    CONFIG["running"] = not CONFIG["running"]
    await save_config()
    schedule_autopost(bot)
    await call.answer("Автопостинг включён" if CONFIG["running"] else "Автопостинг выключен")
    await call.message.edit_text(status_text(), reply_markup=main_menu_kb())


@router.callback_query(AdminFilter(), F.data == "set_frequency")
async def cb_set_frequency(call: CallbackQuery) -> None:
    buttons = [InlineKeyboardButton(text=str(n), callback_data=f"freq:{n}") for n in range(1, 7)]
    kb = InlineKeyboardMarkup(
        inline_keyboard=[buttons, [InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_menu")]]
    )
    await call.answer()
    await call.message.edit_text("Сколько постов в день генерировать? (1-6)", reply_markup=kb)


@router.callback_query(AdminFilter(), F.data.startswith("freq:"))
async def cb_freq_selected(call: CallbackQuery, bot: Bot) -> None:
    n = int(call.data.split(":", 1)[1])
    CONFIG["posts_per_day"] = max(1, min(6, n))
    await save_config()
    if CONFIG["running"]:
        schedule_autopost(bot)
    await call.answer(f"Частота обновлена: {n} пост(ов) в день")
    await call.message.edit_text(status_text(), reply_markup=main_menu_kb())


@router.callback_query(AdminFilter(), F.data == "back_to_menu")
async def cb_back(call: CallbackQuery) -> None:
    await call.answer()
    await call.message.edit_text(status_text(), reply_markup=main_menu_kb())


@router.callback_query(AdminFilter(), F.data == "set_niche")
async def cb_set_niche(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    await state.set_state(NicheStates.waiting_niche)
    await call.message.edit_text(
        f"Текущая ниша: <i>{CONFIG['niche']}</i>\n\nОпишите новую нишу/продукт одним сообщением:"
    )


@router.message(AdminFilter(), NicheStates.waiting_niche)
async def process_niche(message: Message, state: FSMContext) -> None:
    niche = (message.text or "").strip()
    if not niche:
        await message.answer("Пришлите текстовое описание ниши.")
        return
    CONFIG["niche"] = niche[:500]
    await save_config()
    await state.clear()
    await message.answer("✅ Ниша обновлена.", reply_markup=main_menu_kb())


@router.callback_query(AdminFilter(), F.data == "post_now")
async def cb_post_now(call: CallbackQuery, bot: Bot) -> None:
    await call.answer("Генерирую пост...")
    asyncio.create_task(generate_and_send_post(bot))


@router.callback_query(AdminFilter(), F.data == "slideshow_now")
async def cb_slideshow_now(call: CallbackQuery, bot: Bot) -> None:
    await call.answer("Собираю AI-слайдшоу, это займёт пару минут...")
    asyncio.create_task(generate_and_send_slideshow(bot))


@router.callback_query(AdminFilter(), F.data.startswith("publish:"))
async def cb_publish(call: CallbackQuery) -> None:
    token = call.data.split(":", 1)[1]
    item = PENDING.get(token)
    if not item:
        await call.answer("Срок действия истёк или уже обработано", show_alert=True)
        return

    dry_run = not instagram_configured()
    await call.answer("Публикую..." if not dry_run else "Тестовый прогон (Instagram не подключён)...")
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
    status_msg = await call.message.reply(
        "⏳ Загружаю медиа и публикую в Instagram..."
        if not dry_run
        else "⏳ Загружаю медиа на временный хостинг (тестовый прогон)..."
    )

    try:
        if item["kind"] == "carousel":
            media_urls = [await upload_media(p) for p in item["image_paths"]]
        elif item["kind"] == "video":
            media_urls = [await upload_media(item["video_path"])]
        else:
            raise ValueError(f"Неизвестный тип публикации: {item['kind']}")

        if dry_run:
            links = "\n".join(media_urls)
            await status_msg.edit_text(
                "🧪 <b>DRY-RUN</b>: INSTA_ACCOUNT_ID / INSTA_ACCESS_TOKEN не заданы, реальная "
                "публикация пропущена.\n\n"
                "Медиа успешно сгенерировано и загружено на временный хостинг — значит, весь "
                f"пайплайн работает.\nСсылки на файлы:\n{links}\n\n"
                "Добавьте ключи Instagram в переменные окружения, и эта же кнопка начнёт публиковать по-настоящему."
            )
        elif item["kind"] == "carousel":
            media_id = await publish_carousel_to_instagram(media_urls, item["caption"])
            await status_msg.edit_text(f"✅ Карусель опубликована в Instagram! ID медиа: {media_id}")
        else:
            media_id = await publish_reel_to_instagram(media_urls[0], item["caption"])
            await status_msg.edit_text(f"✅ Опубликовано в Instagram! ID медиа: {media_id}")
    except Exception as exc:
        logger.exception("Ошибка публикации")
        await status_msg.edit_text(f"❌ Ошибка публикации: {exc}")
    finally:
        pop_pending(token)
        cleanup_pending_files(item)


@router.callback_query(AdminFilter(), F.data.startswith("cancel:"))
async def cb_cancel(call: CallbackQuery) -> None:
    token = call.data.split(":", 1)[1]
    item = pop_pending(token)
    await call.answer("Отменено")
    try:
        await call.message.edit_reply_markup(reply_markup=None)
    except Exception:
        pass
    await call.message.reply("🔄 Отменено. Черновик удалён.")
    if item:
        cleanup_pending_files(item)


@router.message(AdminFilter(), F.content_type.in_({ContentType.VIDEO, ContentType.DOCUMENT}))
async def handle_video(message: Message, bot: Bot) -> None:
    video = message.video
    document = message.document
    is_video_doc = document is not None and (document.mime_type or "").startswith("video/")
    if video is None and not is_video_doc:
        return

    if video is not None and video.duration and not (30 <= video.duration <= 630):
        await message.reply("⚠️ Пришлите видео длительностью от 1 до 10 минут.")
        return

    file_id = video.file_id if video else document.file_id
    await process_video(bot, message, file_id)


@router.message(AdminFilter())
async def fallback_message(message: Message) -> None:
    await message.answer(status_text(), reply_markup=main_menu_kb())


# --------------------------------------------------------------------------- #
# ЗАПУСК
# --------------------------------------------------------------------------- #


async def on_startup(bot: Bot) -> None:
    await bot.set_my_commands([BotCommand(command="start", description="Открыть меню")])
    schedule_autopost(bot)
    scheduler.add_job(
        cleanup_stale_pending, IntervalTrigger(hours=6), id="cleanup", replace_existing=True
    )
    scheduler.start()
    if CONFIG.get("admin_id"):
        try:
            await bot.send_message(
                CONFIG["admin_id"],
                "🏭 Контент-завод перезапущен и готов к работе.",
                reply_markup=main_menu_kb(),
            )
        except Exception:
            logger.warning("Не удалось уведомить администратора о запуске")


async def on_shutdown(bot: Bot) -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)


async def main() -> None:
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")
