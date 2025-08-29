import os
import numpy as np

from vllm.config import VllmConfig, ModelConfig, SpeculativeConfig
from vllm.v1.spec_decode.static_text_proposer import StaticTextProposer
from vllm.transformers_utils.tokenizer import get_tokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


def make_proposer(k: int = 8) -> tuple[StaticTextProposer, any]:
    mc = ModelConfig(model=MODEL_NAME, trust_remote_code=True)
    sc = SpeculativeConfig(method="static_text", num_speculative_tokens=k)
    cfg = VllmConfig(model_config=mc, speculative_config=sc)
    proposer = StaticTextProposer(cfg)
    tok = get_tokenizer(MODEL_NAME, trust_remote_code=True)
    return proposer, tok


def test_propose_nothing_returns_none():
    proposer, tok = make_proposer(k=8)
    req_id = "t0"
    ctx_ids = np.array([], dtype=np.int32)
    out = proposer.propose(req_id=req_id,
                           context_token_ids=ctx_ids,
                           predicted_token_ids=[])
    assert out is None


def test_exact_match_proposes_next_tokens():
    proposer, tok = make_proposer(k=8)
    req_id = "t1"
    predicted_text = "Alpha\nBeta\nGamma\n"
    pred_ids = tok.encode(predicted_text, add_special_tokens=False)

    # progressively increase context and check proposed ids are the next K
    for prefix_len in [0, 1, min(5, len(pred_ids) - 1)]:
        ctx_ids = np.array(pred_ids[:prefix_len], dtype=np.int32)
        out = proposer.propose(req_id=req_id,
                               context_token_ids=ctx_ids,
                               predicted_token_ids=pred_ids)
        assert out is not None
        expected = pred_ids[prefix_len:prefix_len + 8]
        assert out.tolist() == expected


def test_partial_overlap_missing_line():
    proposer, tok = make_proposer(k=16)
    req_id = "t2"

    predicted_text = (
        "Roses are red\n"
        "Violets are blue\n"
        "Sugar is sweet\n"
        "And so are you\n"
    )
    pred_ids = tok.encode(predicted_text, add_special_tokens=False)

    # Context skips the second line entirely, then starts the final line.
    context_text = (
        "Roses are red\n"
        "Sugar is sweet\n"
        "And so are"
    )
    ctx_ids = np.array(tok.encode(context_text, add_special_tokens=False),
                       dtype=np.int32)

    out = proposer.propose(req_id=req_id,
                           context_token_ids=ctx_ids,
                           predicted_token_ids=pred_ids)

    # Should propose continuation of the last line (" you\n...")
    assert out is not None and len(out) > 0
    proposed_text = tok.decode(out.tolist())
    assert proposed_text.startswith(" you\n")


def test_partial_overlap_with_extra_line_then_realign_on_newline():
    proposer, tok = make_proposer(k=16)
    req_id = "t3"

    predicted_text = (
        "Roses are red\n"
        "Violets are blue\n"
        "Sugar is sweet\n"
        "And so are you\n"
    )
    pred_ids = tok.encode(predicted_text, add_special_tokens=False)

    # First provide context that includes an extra unmatched line (no newline at end)
    context_text_part = (
        "Roses are red\n"
        "EXTRA LINE THAT DOES NOT MATCH"
    )
    ctx_ids_part = np.array(
        tok.encode(context_text_part, add_special_tokens=False), dtype=np.int32)

    out1 = proposer.propose(req_id=req_id,
                            context_token_ids=ctx_ids_part,
                            predicted_token_ids=pred_ids)
    # While mid-line and misaligned: should not propose yet
    assert out1 is None

    # Now complete the extra line and add a matching next predicted line
    context_text = context_text_part + "\nViolets are blue\n"
    ctx_ids = np.array(tok.encode(context_text, add_special_tokens=False),
                       dtype=np.int32)

    out2 = proposer.propose(req_id=req_id,
                            context_token_ids=ctx_ids,
                            predicted_token_ids=pred_ids)
    # After newline and realignment, should propose next predicted line
    assert out2 is not None and len(out2) > 0
    proposed_text = tok.decode(out2.tolist())
    assert proposed_text.startswith("Sugar is sweet\n")
