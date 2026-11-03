"""Public aliases for the selected Qwen and InternVL3 EVA configurations."""
import os
from functools import partial

TOOLS = ['ocr', 'GroundingDINO', 'DrawLine', 'DrawShape', 'GetBarInfo']
VARIANTS = {
    'HallusionBench': ('EVA-Qwen-HallusionBench', 'avg_mid1152_obs12'),
    'MMMU_DEV_VAL': ('EVA-Qwen-MMMU', 'r_plain'),
    'MMStar': ('EVA-Qwen-MMStar', 'v2_answer_first'),
    'POPE': ('EVA-Qwen-POPE', 'c_native1280'),
}
INTERNVL_VARIANTS = {
    'HallusionBench': ('EVA-InternVL3-HallusionBench', 'h_symmetric12'),
    'MMMU_DEV_VAL': ('EVA-InternVL3-MMMU', 'baseline_control'),
    'MMStar': ('EVA-InternVL3-MMStar', 's2_exam_sparse'),
    'POPE': ('EVA-InternVL3-POPE', 'p3_gate13'),
}
VARIANTS_BY_BACKBONE = {'qwen': VARIANTS, 'internvl3': INTERNVL_VARIANTS}


def model_registry(backbone=None):
    if backbone not in (None, 'qwen', 'internvl3'):
        raise ValueError(f'Unknown backbone: {backbone}')
    models = {}
    if backbone in (None, 'qwen'):
        models.update(qwen_registry())
    if backbone in (None, 'internvl3'):
        models.update(internvl_registry())
    return models


def qwen_registry():
    from vlmeval.vlm.eva_tuning import Qwen2VLTuningToolAgent
    from vlmeval.vlm.eva_mmmu_followup import Qwen2VLMMMUFollowupToolAgent
    from vlmeval.vlm.eva_mmstar_followup import Qwen2VLMMStarFollowupToolAgent
    from vlmeval.vlm.eva_pope_followup import Qwen2VLPOPEFollowupToolAgent

    common = dict(
        model_path=os.environ.get('EVA_MODEL_PATH', 'Qwen/Qwen2.5-VL-7B-Instruct'),
        min_pixels=1003520, max_pixels=12845056, use_custom_prompt=False,
        controller_url=os.environ.get('EVA_CONTROLLER_URL', 'http://localhost:20001'),
        tools=TOOLS, save_intermediate=True, max_tool_rounds=2,
        max_new_tokens=2048, top_p=0.001, top_k=1, temperature=0.01,
        repetition_penalty=1.0, do_sample=True,
        torch_dtype=os.environ.get('EVA_TORCH_DTYPE', 'auto'),
        attn_implementation=os.environ.get('EVA_ATTN_IMPLEMENTATION', 'flash_attention_2'),
    )
    return {
        'EVA-Qwen-HallusionBench': partial(Qwen2VLTuningToolAgent, tuning_profile='avg_mid1152_obs12', **common),
        'EVA-Qwen-MMMU': partial(Qwen2VLMMMUFollowupToolAgent, benchmark_variant='r_plain', **common),
        'EVA-Qwen-MMStar': partial(Qwen2VLMMStarFollowupToolAgent, benchmark_variant='v2_answer_first', **common),
        'EVA-Qwen-POPE': partial(Qwen2VLPOPEFollowupToolAgent, benchmark_variant='c_native1280', **common),
    }


def internvl_registry():
    from vlmeval.vlm.eva_internvl_hallusion_followup import InternVL3HallusionFollowupToolAgent
    from vlmeval.vlm.eva_internvl_mmmu_followup import InternVL3MMMUFollowupToolAgent
    from vlmeval.vlm.eva_internvl_mmstar_round2 import InternVL3MMStarRound2ToolAgent
    from vlmeval.vlm.eva_internvl_pope_round3 import InternVL3POPERound3ToolAgent

    common = dict(
        model_path=os.environ.get('EVA_INTERNVL_MODEL_PATH', 'OpenGVLab/InternVL3-14B'),
        controller_url=os.environ.get('EVA_CONTROLLER_URL', 'http://localhost:20001'),
        use_custom_prompt=False, save_intermediate=True, max_tool_rounds=2,
        base_max_tiles=7, total_max_tiles=28, max_new_tokens=2048, do_sample=False,
    )
    return {
        'EVA-InternVL3-HallusionBench': partial(InternVL3HallusionFollowupToolAgent, benchmark_variant='h_symmetric12', **common),
        'EVA-InternVL3-MMMU': partial(InternVL3MMMUFollowupToolAgent, benchmark_variant='baseline_control', **common),
        'EVA-InternVL3-MMStar': partial(InternVL3MMStarRound2ToolAgent, benchmark_variant='s2_exam_sparse', **common),
        'EVA-InternVL3-POPE': partial(InternVL3POPERound3ToolAgent, benchmark_variant='p3_gate13', **common),
    }
