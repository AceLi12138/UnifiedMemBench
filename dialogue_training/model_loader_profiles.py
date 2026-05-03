from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple


@dataclass(frozen=True)
class ModelLoaderProfile:
    name: str
    model_types: Tuple[str, ...] = ()
    path_basenames: Tuple[str, ...] = ()
    path_keywords: Tuple[str, ...] = ()
    config_aliases: Tuple[Tuple[str, str], ...] = ()
    class_attr_defaults: Tuple[Tuple[str, str], ...] = ()
    auto_model_class_name: str = "AutoModelForCausalLM"
    trust_remote_code: bool = True
    omit_temperature_when_greedy: bool = False
    use_generation_config_eos_token_id: bool = False
    repetition_penalty: float | None = None
    device_map: Any = "auto"
    post_load_device: str | None = None


DEFAULT_MODEL_LOADER_PROFILE = ModelLoaderProfile(name="default")

MODEL_LOADER_PROFILES: Tuple[ModelLoaderProfile, ...] = (
    ModelLoaderProfile(
        name="glm_hf",
        path_basenames=("glm-4-9b-chat-1m-hf",),
        auto_model_class_name="AutoModelForCausalLM",
        trust_remote_code=False,
        omit_temperature_when_greedy=True,
        use_generation_config_eos_token_id=True,
        repetition_penalty=1.05,
        device_map="auto",
        post_load_device=None,
    ),
    ModelLoaderProfile(
        name="chatglm_custom_code",
        path_basenames=("glm-4-9b-chat-1m",),
        config_aliases=(("max_length", "seq_length"),),
        class_attr_defaults=(("all_tied_weights_keys", "empty_dict"), ("_tp_plan", "empty_list")),
        auto_model_class_name="AutoModelForCausalLM",
        trust_remote_code=True,
        device_map=None,
        post_load_device="cuda",
    ),
)

_CLASS_ATTR_DEFAULT_FACTORIES = {
    "empty_dict": dict,
    "empty_list": list,
}


def _ensure_generation_mixin(model_class: Any) -> tuple[Any, bool]:
    from transformers import GenerationMixin

    if issubclass(model_class, GenerationMixin):
        return model_class, False

    patched_class = type(
        f"{model_class.__name__}WithGenerationMixin",
        (model_class, GenerationMixin),
        {},
    )
    return patched_class, True


def _load_auto_map_from_model_path(model_name_or_path: str) -> dict[str, str]:
    model_path = Path(str(model_name_or_path or "")).expanduser()
    if not model_path.exists() or not model_path.is_dir():
        return {}
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    auto_map = payload.get("auto_map")
    return auto_map if isinstance(auto_map, dict) else {}


def resolve_model_loader_profile(config: Any, model_name_or_path: str) -> ModelLoaderProfile:
    model_type = str(getattr(config, "model_type", "") or "").strip().lower()
    path_text = str(model_name_or_path or "").strip().lower()
    path_basename = Path(path_text).name.lower()

    for profile in MODEL_LOADER_PROFILES:
        if path_basename and path_basename in profile.path_basenames:
            return profile
        if model_type and model_type in profile.model_types:
            return profile
        if path_text and any(keyword in path_text for keyword in profile.path_keywords):
            return profile
    return DEFAULT_MODEL_LOADER_PROFILE


def prepare_config_for_model_loading(
    config: Any,
    model_name_or_path: str,
) -> Tuple[Any, ModelLoaderProfile, List[Tuple[str, str]]]:
    profile = resolve_model_loader_profile(config, model_name_or_path)
    applied_aliases: List[Tuple[str, str]] = []

    for target_attr, source_attr in profile.config_aliases:
        if hasattr(config, target_attr) or not hasattr(config, source_attr):
            continue
        setattr(config, target_attr, getattr(config, source_attr))
        applied_aliases.append((target_attr, source_attr))

    return config, profile, applied_aliases


def prepare_model_class_for_loading(
    config: Any,
    model_name_or_path: str,
    auto_model_class_name: str,
) -> Tuple[Any | None, ModelLoaderProfile, List[str]]:
    profile = resolve_model_loader_profile(config, model_name_or_path)
    auto_map = getattr(config, "auto_map", None) or _load_auto_map_from_model_path(model_name_or_path) or {}
    class_reference = auto_map.get(auto_model_class_name)
    if not class_reference or not profile.class_attr_defaults:
        return None, profile, []

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    applied_attrs: List[str] = []
    model_class = get_class_from_dynamic_module(class_reference, model_name_or_path)
    if profile.name == "chatglm_custom_code":
        model_class, wrapped = _ensure_generation_mixin(model_class)
        if wrapped:
            applied_attrs.append("generation_mixin")
    for attr_name, default_kind in profile.class_attr_defaults:
        if hasattr(model_class, attr_name):
            current_value = getattr(model_class, attr_name)
            if current_value is not None:
                continue
        factory = _CLASS_ATTR_DEFAULT_FACTORIES[default_kind]
        setattr(model_class, attr_name, factory())
        applied_attrs.append(attr_name)
    return model_class, profile, applied_attrs


def prepare_model_instance_for_loading(
    model: Any,
    model_name_or_path: str,
) -> Tuple[Any, ModelLoaderProfile, List[str]]:
    profile = resolve_model_loader_profile(getattr(model, "config", None), model_name_or_path)
    applied_patches: List[str] = []

    if profile.name != "chatglm_custom_code":
        return model, profile, applied_patches

    patched_class, wrapped = _ensure_generation_mixin(model.__class__)
    if wrapped:
        model.__class__ = patched_class
        applied_patches.append("generation_mixin")

    if not hasattr(model, "generation_config"):
        from transformers import GenerationConfig

        model.generation_config = GenerationConfig.from_model_config(model.config)
        applied_patches.append("generation_config")

    return model, profile, applied_patches
