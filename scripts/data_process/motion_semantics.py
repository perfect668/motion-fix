"""Canonical filename-derived semantics for the motion data pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


CATEGORY_DESCRIPTIONS = {
    "object_manipulation_carry": "carry, hold, pick/place, tool and prop interactions",
    "locomotion_walk": "walking clips, side steps, walking loops, starts and stops",
    "locomotion_jog_run": "jogging and running clips, loops, starts and stops",
    "jump": "jumps, high jumps, reach jumps and jump turns",
    "dance": "dance and dancing routines",
    "ground_low_posture": "sit, kneel, crawl, lie and on-ground motions",
    "turn_transition": "turns, stance changes and step-rotate transitions",
    "idle_stance": "idle, stance, relaxed and looking-around standing clips",
    "upper_body_gesture": "standing gestures, reaching, clapping, saluting and similar upper-body actions",
    "obstacle_contact_avoidance": "obstacle avoidance, bumps, collisions and body checks",
    "injury_impaired_gait": "injured or impaired gait variants",
    "exercise_sport": "exercise, sport-like and training motions",
    "daily_social_expression": "daily-life gestures, emotions and social expressions",
    "kick_throw_stoop": "kick, throw, stoop and similar short full-body actions",
    "other": "uncategorized filename patterns",
}

HIGH_DYNAMIC_CATEGORIES = {
    "jump",
    "dance",
    "locomotion_jog_run",
    "exercise_sport",
    "obstacle_contact_avoidance",
    "kick_throw_stoop",
}

HIGH_DYNAMIC_KEYWORDS = (
    "jump",
    "hop",
    "dance",
    "dancing",
    "jog",
    "run",
    "sprint",
    "kick",
    "throw",
    "burpee",
    "combat",
    "obstacle",
    "bump",
)

EXTERNAL_SUPPORT_PHRASES = (
    "against_wall",
    "brace_against",
    "bracing_against",
    "lean_against",
    "lean_on",
    "leaning_on",
    "leaning_wall",
    "jump_off_wall",
    "nailing_wall",
    "off_wall",
    "prop_against",
    "propped_against",
    "rest_on",
    "resting_on",
    "supported_by",
    "wall_lean",
)
EXTERNAL_SUPPORT_ACTION_TOKENS = {
    "brace",
    "bracing",
    "hang",
    "hanging",
    "lean",
    "leaning",
    "prop",
    "propped",
    "propping",
    "rest",
    "resting",
    "support",
    "supported",
    "supporting",
}
EXTERNAL_SUPPORT_OBJECT_TOKENS = {
    "bar",
    "chair",
    "counter",
    "door",
    "fence",
    "ladder",
    "pole",
    "rail",
    "railing",
    "table",
    "wall",
}

DANCE_TOKENS = ("dance", "dancing", "hiphop", "vouge", "retro", "latino", "macarena", "western")
JUMP_TOKENS = ("jump", "jumping", "hop", "hopping")
RUN_TOKENS = ("jog", "run", "running", "sprint")


@dataclass(frozen=True)
class MotionSemantics:
    motion_family: str
    normalized_family: str
    actor: str
    is_mirror: bool
    category: str
    dynamic_group: str
    external_support_dependency: bool


def normalize_motion_text(value):
    return "".join(ch if ch.isalnum() else "_" for ch in str(value).lower())


def motion_family_from_path(value):
    return Path(value).stem.split("__", 1)[0]


def normalized_family(name):
    tokens = str(name).lower().split("_")
    while tokens and (tokens[-1].isdigit() or re.fullmatch(r"v[0-9]+", tokens[-1])):
        tokens.pop()
    while tokens and tokens[-1] in {"r", "l", "left", "right"}:
        tokens.pop()
    return "_".join(tokens) if tokens else str(name).lower()


def actor_id_from_path(value):
    match = re.search(r"__(A[0-9]+)", Path(value).stem)
    return match.group(1) if match else "unknown"


def is_mirror_path(value):
    return Path(value).stem.endswith("_M")


def has_external_support_dependency(value):
    text = normalize_motion_text(value)
    if any(phrase in text for phrase in EXTERNAL_SUPPORT_PHRASES):
        return True
    tokens = {token for token in text.split("_") if token}
    return bool(tokens & EXTERNAL_SUPPORT_ACTION_TOKENS) and bool(
        tokens & EXTERNAL_SUPPORT_OBJECT_TOKENS
    )


def _has_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def categorize_motion(name):
    text = str(name).lower()
    if _has_any(text, ("bump", "obstacle", "body_check", "avoid_")):
        return "obstacle_contact_avoidance"
    if _has_any(text, ("jump", "hop")):
        return "jump"
    if _has_any(text, ("dance", "dancing", "mohak")):
        return "dance"
    if _has_any(text, ("kneel", "sit", "crawl", "lie", "on_ground", "balled_up")):
        return "ground_low_posture"
    if _has_any(text, ("injured", "inj_")):
        return "injury_impaired_gait"
    if _has_any(
        text,
        (
            "one_hand", "two_hands", "pick_up", "put_down", "hold", "carry", "crate",
            "box", "bucket", "big_", "small_", "medium_", "heavy", "light", "tool",
            "axe", "saw", "broom", "mop", "watering", "painting", "operating", "item",
            "trash", "apple", "binoculars",
        ),
    ):
        return "object_manipulation_carry"
    if _has_any(text, ("walk", "sideway_walk", "loop_forward_walk", "loop_backward_walk")):
        return "locomotion_walk"
    if _has_any(text, ("jog", "run", "sprint")):
        return "locomotion_jog_run"
    if _has_any(text, ("turn", "step_rotate", "stance_change", "change_right", "change_left")):
        return "turn_transition"
    if _has_any(text, ("idle", "stance", "relax", "looking_around")):
        return "idle_stance"
    if _has_any(text, ("exercise", "burpee", "ab_bicycle", "push_up", "sport", "training")):
        return "exercise_sport"
    if _has_any(
        text,
        (
            "clap", "salute", "reach", "reaching", "checking_time", "thinking", "confusion",
            "welcoming", "pocket_searching", "itching", "chefs_kiss", "omg", "don_t_know",
            "no_see", "no_hear", "fixing_something", "brush", "dust", "body_search",
            "body_stretch", "rubbing", "wiping", "show_bicep", "praying", "listening",
            "clearing_ear", "yawn", "sneeze", "bow", "beckon", "greeting", "bye", "wave",
        ),
    ):
        return "upper_body_gesture"
    if _has_any(text, ("kick", "throw", "stoop")):
        return "kick_throw_stoop"
    if _has_any(
        text,
        (
            "triumph", "victory", "crowd", "screaming", "lamenting", "puke", "eureka",
            "angry", "alone", "bravo", "calm_down", "as_you_wish", "maybe", "tasty",
            "just_realised", "hurry", "eating", "drinking", "smoke", "stinky", "sweat",
            "freezing_cold", "horse_riding", "lasso",
        ),
    ):
        return "daily_social_expression"
    return "other"


def dynamic_group_from_text(value, category=""):
    text = normalize_motion_text(value)
    if any(token in text for token in DANCE_TOKENS):
        return "dance"
    if any(token in text for token in JUMP_TOKENS):
        return "jump"
    if any(token in text for token in RUN_TOKENS) or category == "locomotion_jog_run":
        return "run_jog"
    if category in {"exercise_sport", "kick_throw_stoop", "obstacle_contact_avoidance"}:
        return "sport_kick_obstacle"
    if category in {"jump", "dance"}:
        return category
    return "other"


def is_high_dynamic(value, category=""):
    text = normalize_motion_text(value)
    return category in HIGH_DYNAMIC_CATEGORIES or any(keyword in text for keyword in HIGH_DYNAMIC_KEYWORDS)


def describe_motion(value):
    family = motion_family_from_path(value)
    category = categorize_motion(family)
    return MotionSemantics(
        motion_family=family,
        normalized_family=normalized_family(family),
        actor=actor_id_from_path(value),
        is_mirror=is_mirror_path(value),
        category=category,
        dynamic_group=dynamic_group_from_text(family, category),
        external_support_dependency=has_external_support_dependency(value),
    )
