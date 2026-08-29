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
from PIL import Image, ImageDraw, ImageFont, ImageOps

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

ASS_HEADER_TEMPLATE = """[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 2
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
техзадание фотографу, а не общее описание. ОБЯЗАТЕЛЬНО включи все пункты подряд, в одном
предложении-рецепте:
1. Конкретный человек и его ДЕЙСТВИЕ прямо сейчас (не "человек с ноутбуком", а "a woman
   mid-laugh pointing at her phone screen while leaning against a kitchen counter") + видимая
   эмоция на лице (excited, relieved, surprised, focused — не "happy" вообще).
2. Конкретное место и время суток/освещение (golden hour through a window, warm string lights
   at dusk, overcast soft daylight, neon sign glow at night) — не абстрактный "natural light".
3. Операторские детали: "shot on 35mm film, shallow depth of field, bokeh background", или
   "shot on iPhone, slightly overexposed, candid angle" — конкретный визуальный стиль, разный
   для каждого слайда карусели (не повторяй одну и ту же формулировку 5 раз).
4. Композиция: "close-up", "over-the-shoulder shot", "wide environmental shot", "from above" —
   меняй ракурс от слайда к слайду, чтобы карусель не выглядела как один и тот же кадр.
Обязательно заверши каждый image_prompt фразой: "photorealistic, editorial photography, high
detail, no text, no watermark, no logo, no deformed hands".
Плохой пример (так не пиши): "A person working on a laptop, natural light, professional photo".
Хороший пример: "A young man in a hoodie throwing his hands up in triumph at his kitchen table,
laptop screen glowing on his surprised face, golden hour sunset light through the window behind
him, shot on 35mm film with warm grain, shallow depth of field, low angle shot, photorealistic,
editorial photography, high detail, no text, no watermark, no logo, no deformed hands".

Также для каждого слайда пиши "hook_text": короткая цепляющая фраза на русском (3-8 слов) для
крупной надписи ПОВЕРХ фото, как в вирусных Reels/каруселях — продолжает мысль по формуле
Hook-Story-Offer для этого слайда.

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
        max_tokens=2000,
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
    encoded_prompt = quote(
        f"{prompt}, photorealistic, candid lifestyle photography, natural light, 8k, sharp focus, "
        "no text, no watermark, no logo"
    )
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


def draw_hook_text(image: Image.Image, text: str) -> Image.Image:
    width, height = image.size
    font_size = 72
    font = ImageFont.truetype(str(ASSETS_DIR / "DejaVuSans-Bold.ttf"), font_size)
    lines = _wrap_text(text, font, width - 120)

    line_height = font_size + 16
    block_height = line_height * len(lines) + 70

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(
        [0, height - block_height, width, height], fill=(0, 0, 0, 165)
    )
    composed = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(composed)

    y = height - block_height + 35
    for line in lines:
        bbox = font.getbbox(line)
        x = (width - (bbox[2] - bbox[0])) / 2
        _draw_outlined_text(draw, (x, y), line, font)
        y += line_height

    return composed.convert("RGB")


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


async def generate_slide_image(image_prompt: str, hook_text: str) -> Path:
    image = await fetch_flux_image(image_prompt)
    final_path = TEMP_DIR / f"slide_{uuid.uuid4().hex}.jpg"

    def _draw_and_save() -> None:
        framed = draw_hook_text(image, hook_text)
        framed.save(final_path, format="JPEG", quality=92)

    await asyncio.to_thread(_draw_and_save)
    return final_path


async def generate_carousel_slides(slides: list[dict]) -> list[Path]:
    tasks = [
        asyncio.create_task(generate_slide_image(slide["image_prompt"], slide["hook_text"]))
        for slide in slides
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


async def pick_viral_segment(niche: str, segments: list[dict], duration: float) -> dict:
    lines = [f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text'].strip()}" for seg in segments]
    transcript_block = "\n".join(lines)
    system_prompt = (
        "Ты — вирусный видеоредактор и SMM-стратег, который находит самые цепляющие "
        "20-50-секундные фрагменты в длинных видео для Reels. Отвечай строго валидным JSON."
    )
    user_prompt = f"""
Ниша: {niche}
Общая длительность видео: {duration:.1f} сек.

Транскрипт с таймкодами (секунды):
{transcript_block}

Выбери ОДИН самый вирусный, экспертный или эмоциональный непрерывный фрагмент длиной от 20 до 50 секунд.
Тайминги start и end должны попадать в пределы видео (0..{duration:.1f}) и совпадать с границами реплик
из транскрипта.
Также напиши:
- "caption": продающий кэпшн на русском с сильным хуком в первой строке и призывом к действию,
  до 900 символов, с эмодзи и 3-5 хэштегами (это текст ПОД видео в Instagram).
- "hook_text": короткая цепляющая фраза на русском (3-7 слов, БЕЗ хэштегов и эмодзи) — она будет
  крупной надписью поверх первых секунд самого видео, чтобы остановить скролл. Не дублируй
  дословно первую строку caption, сформулируй ударнее и короче.

Верни строго JSON: {{"start": <число секунд>, "end": <число секунд>, "caption": "<текст>", "hook_text": "<текст>"}}
"""
    completion = await groq_client.chat.completions.create(
        model=GROQ_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=1200,
        response_format={"type": "json_object"},
    )
    data = extract_json(completion.choices[0].message.content)
    if "hook_text" not in data:
        raise ValueError(f"Некорректный ответ LLM: {data}")
    start = max(0.0, float(data["start"]))
    end = min(duration, float(data["end"]))
    if end - start < 5:
        raise ValueError("LLM вернул слишком короткий фрагмент")
    if end - start > 60:
        end = start + 60
    return {"start": start, "end": end, "caption": data["caption"], "hook_text": data["hook_text"]}


HOOK_OVERLAY_SECONDS = 2.5


async def render_reel(source: Path, ass_path: Path, start: float, end: float, hook_text: str) -> Path:
    output_path = TEMP_DIR / f"reel_{uuid.uuid4().hex}.mp4"
    subtitles_arg = escape_ffmpeg_filter_path(str(ass_path))
    fontsdir_arg = escape_ffmpeg_filter_path(str(ASSETS_DIR))
    # fontsdir указывает на шрифт, вложенный прямо в репозиторий (assets/), чтобы не зависеть
    # от системных шрифтов хостинга. Сам стиль (шрифт/цвет/отступы) задан внутри .ass-файла
    # (build_ass), а не через force_style — см. комментарий там про баг с масштабированием.
    # crop-to-fill в 9:16 независимо от ориентации исходника: если видео шире цели — обрезаем
    # по бокам (crop по ширине), если уже уже цели (портретная/квадратная съёмка, как часто
    # бывает с видео "сверху вниз") — обрезаем сверху/снизу (crop по высоте). Без этого условия
    # ffmpeg пытался обрезать по ширине даже там, где исходник уже; scale потом растягивал
    # результат и картинка искажалась.
    crop_expr = (
        "crop="
        "'if(gte(iw/ih,9/16),ih*9/16,iw)':"
        "'if(gte(iw/ih,9/16),ih,iw*16/9)'"
    )

    hook_png = await asyncio.to_thread(render_hook_overlay_png, hook_text)
    try:
        filter_complex = (
            f"[0:v]{crop_expr},scale=1080:1920,"
            f"subtitles='{subtitles_arg}':fontsdir='{fontsdir_arg}'[base];"
            f"[base][1:v]overlay=0:0:enable='between(t,0,{HOOK_OVERLAY_SECONDS})'[outv]"
        )
        await run_ffmpeg(
            [
                FFMPEG_BIN, "-y",
                "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                "-i", str(source),
                "-i", str(hook_png),
                "-filter_complex", filter_complex,
                "-map", "[outv]", "-map", "0:a",
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
    ass_path: Optional[Path] = None
    reel_path: Optional[Path] = None

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

        await status.edit_text("🧠 Ищу самый вирусный фрагмент и пишу кэпшн...")
        pick = await pick_viral_segment(CONFIG["niche"], segments, duration)

        await status.edit_text(
            f"✂️ Нарезаю {pick['start']:.0f}-{pick['end']:.0f} сек, кроплю в 9:16, жгу субтитры..."
        )
        ass_path = TEMP_DIR / f"sub_{uuid.uuid4().hex}.ass"
        ass_path.write_text(build_ass(words, pick["start"], pick["end"]), encoding="utf-8")

        reel_path = await render_reel(raw_path, ass_path, pick["start"], pick["end"], pick["hook_text"])

        token = register_pending("video", video_path=str(reel_path), caption=pick["caption"])
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="🚀 Опубликовать в Instagram", callback_data=f"publish:{token}"
                    ),
                    InlineKeyboardButton(text="🔄 Отмена / Переделать", callback_data=f"cancel:{token}"),
                ]
            ]
        )
        await status.delete()
        await bot.send_video(
            message.chat.id,
            FSInputFile(reel_path),
            caption=pick["caption"][:1024],
            reply_markup=kb,
            supports_streaming=True,
        )
    except Exception as exc:
        logger.exception("Ошибка обработки видео")
        await status.edit_text(f"❌ Ошибка обработки видео: {exc}")
        if reel_path and os.path.exists(reel_path):
            os.remove(reel_path)
    finally:
        for path in (raw_path, audio_path, ass_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


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
