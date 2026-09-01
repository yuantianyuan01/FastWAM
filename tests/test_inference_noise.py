import torch

from fastwam.models.wan22.noise import make_inference_generator


def _sample_joint_noise(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    action = torch.randn(
        2048,
        generator=make_inference_generator(seed, "cpu"),
    )
    video = torch.randn(
        2048,
        generator=make_inference_generator(seed, "cpu", stream_offset=1),
    )
    return action, video


def test_joint_noise_uses_reproducible_distinct_streams():
    action, video = _sample_joint_noise(42)
    repeated_action, repeated_video = _sample_joint_noise(42)

    torch.testing.assert_close(action, repeated_action, rtol=0, atol=0)
    torch.testing.assert_close(video, repeated_video, rtol=0, atol=0)
    assert not torch.equal(action, video)


def test_action_stream_preserves_action_only_seed_behavior():
    seed = 42
    action = torch.randn(
        128,
        generator=make_inference_generator(seed, "cpu"),
    )
    action_only = torch.randn(
        128,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )

    torch.testing.assert_close(action, action_only, rtol=0, atol=0)
