# Security

Report security issues privately to the repository maintainers through your existing Dobot
contact or the hosting platform's private vulnerability-reporting channel. Include the affected
revision, a minimal reproduction, and the deployment context. Do not post credentials or
private robot observations in public issues.

## Deployment boundaries

- Robot TCP v2 uses msgpack, mutual nonce authentication, sequence numbers and HMAC-SHA256.
  Set the same random `NEXUS_ROBOT_TOKEN` (at least 32 bytes) on both hosts. There is no pickle
  fallback. Both hosts must use matching v2 scripts. Frames are limited to 32 MiB and have a
  30-second deadline once their first byte arrives. The authenticated client may wait
  for an operator indefinitely between observations; handshakes and action replies still
  time out after 30 seconds. Authentication does not encrypt camera or joint data: use an encrypted
  tunnel or VPN across untrusted networks.
- HTTP and WebSocket servers bind to loopback by default. Binding another interface requires
  `NEXUS_POLICY_TOKEN` (at least 32 bytes); clients send `Authorization: Bearer <token>`.
  `/healthz` is public. Keep service tokens in environment variables or a secret manager.
  Use a TLS gateway or encrypted tunnel when crossing untrusted networks.
- HTTP `/sessions` creates a random session capability. Keep its `session_id` private and use
  it with `/act`, `/reset`, and `DELETE /sessions/{session_id}`. Sessions expire after 15 minutes
  idle and are limited to 16. HTTP requests are limited to 8 MiB and sampling to 100 steps.
  The model is serialized across requests; tasks and environment identities require a reset.
- WebSocket requests and responses are limited to 32 MiB. Memory history grows on demand,
  with per-connection environment and byte quotas. See `docs/pi05_memory.md` for configuration.
- All public action outputs and exported weights must be finite. These checks do not replace
  embodiment-specific joint, velocity, collision, emergency-stop, and operator safeguards.

## Files and model code

Deployment `.pt` and training RNG/trainer sidecars use `torch.load(weights_only=True)`;
`.safetensors` is also supported. There is no automatic unsafe loading retry. Unrestricted
legacy checkpoints should be converted in a separate, isolated, trusted environment.

PyTorch DCP resume metadata uses Python pickle internally. Exact resume is disabled unless
`train.trust_resume_checkpoint=true` is explicitly set. Enable it only for checkpoints from
your own training runs in storage you control. Training state v3 stores restricted-loadable
RNG data; earlier training states require a new run from deployment weights.

Model targets in configuration are Python import paths and execute model code. The legacy
collection-data converter reads pickle input. Treat model configurations, Python plugins,
DCP checkpoints and collection pickle files as executable trusted inputs. Do not accept them
from unauthenticated uploads. Tensor loading is not a resource sandbox: constrain process
memory, file sizes and GPU allocation when handling external model assets.
