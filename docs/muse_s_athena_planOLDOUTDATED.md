# Muse S Athena Integration Plan (outdated)

## Sources reviewed
- Muse official comparison post “Athena vs. Muse S G2: The biggest brain tech upgrade yet” (2025-03-21) – https://choosemuse.com/blogs/news/athena-vs-muse-s-g2-the-biggest-brain-tech-upgrade-yet
- Cybernews hands-on review (2025-05-21) highlighting Athena sensor stack (EEG + fNIRS, upgraded PPG, IMU) – https://cybernews.com/health-tech/muse-s-review/
- BrainFlow docs “Supported Boards -> Muse S BLED” – https://brainflow.readthedocs.io/en/stable/SupportedBoards.html (shows BoardIds.MUSE_S_BLED_BOARD, BLED112 dongle requirement, presets for EEG/IMU/PPG, config flags p50/p61)

## Device capability summary
- EEG: four on-head channels plus four auxiliary channels; higher-resolution ADC (14-bit vs 12-bit in Muse S G2) for cleaner signal and lower noise floor (per Muse blog). Historical Muse S devices stream EEG at 256 Hz; confirm Athena SDK sampling rate once released.
- fNIRS: five optodes on the frontal cortex, 64 Hz sampling (per Muse blog). Measures oxygenated blood changes to estimate cognitive workload.
- PPG/heart: triple-wavelength (IR/NIR/Red) sensor upgraded to 20-bit resolution (Muse blog) for HR/HRV; sample rate to confirm with SDK.
- Motion: accelerometer and gyroscope (noted in Cybernews review) for movement/pose tracking and artifact detection.
- Connectivity: BLE 5.3 (Muse blog) for lower latency and better throughput; USB-C charging.
- Form factor: silver-thread fabric EEG electrodes (Muse blog) for better contact durability.

## BrainFlow viability for Athena
- Current BrainFlow support list includes Muse S BLED (Gen 2) via BoardIds.MUSE_S_BLED_BOARD and requires a BLED112 dongle. No Athena-specific board ID is published yet.
- Hypothesis: Athena may maintain Muse S BLE characteristics and work through BLED112 + Muse S board profile. Needs empirical validation; if not, a BrainFlow board definition update is required.
- BrainFlow streaming features we can leverage:
  - DEFAULT_PRESET: EEG; enable 5th EEG channel via `board.config_board("p50")`.
  - AUXILIARY_PRESET: accelerometer + gyroscope (enabled by default).
  - ANCILLARY_PRESET: PPG; enable via `board.config_board("p61")`.
- Missing in BrainFlow today: fNIRS support. We will likely only get EEG + IMU + PPG through BrainFlow unless/until fNIRS is added.

## Fit with current codebase
- Implement as a concrete `EEGDeviceInterface` (backend/eeg/device_interface.py) device class to align with existing simulator/other hardware abstractions.
- Expose configuration and lifecycle through the FastAPI service (backend/api/main.py) and websocket streaming layer so downstream audio/visual pipelines can consume EEG + auxiliary modalities.
- Keep sampling-rate metadata, channel naming, and modality availability surfaced in device `get_info()` for API clients and UI to adapt.

## Proposed architecture
- Driver/adapter
  - New class `MuseSAthenaDevice(EEGDeviceInterface)` under `backend/eeg/devices/muse_s_athena.py`.
  - Responsibilities: BLE discovery/pairing, authentication (if required by Muse SDK), stream subscription (EEG, fNIRS, PPG, IMU), buffering, health monitoring, graceful reconnect.
  - Dependency options: prefer official Muse SDK if Python bindings exist; otherwise use `bleak` to implement BLE GATT client with Muse developer characteristics (to be pulled from SDK docs).
- Data model and synchronization
  - Maintain per-modality ring buffers with timestamps; expose `read_samples(n)` returning synchronized EEG (and optionally fNIRS/PPG/IMU) slices.
  - Implement downsampling/upsampling utilities to align 256 Hz EEG with 64 Hz fNIRS and slower PPG; choose a common processing rate (e.g., 128 Hz for EEG pipelines, 32–64 Hz for fNIRS/PPG) and provide resampling hooks.
  - Channel metadata: EEG labels (e.g., TP9, AF7, AF8, TP10 plus AUX1–AUX4), fNIRS optode IDs (O1–O5), PPG wavelengths (IR/NIR/Red), IMU axes.
- Configuration surface (via settings/env)
  - BLE device name/ID whitelist, connection timeout/retry limits.
  - Per-modality enable flags (e.g., `ENABLE_FNIRS`, `ENABLE_PPG`), sampling targets, and buffer durations.
  - Signal-quality thresholds (contact quality for EEG electrodes, fNIRS saturation limits, PPG confidence).
  - Logging levels and diagnostic event forwarding.
- API/Websocket integration
  - Add device selection/config endpoints (e.g., `/api/v1/devices` GET/POST) or extend existing session creation to accept `device_type=muse_s_athena` and modality flags.
  - Stream payload schema additions: include per-sample timestamps, modality tags, and quality metrics; ensure backward compatibility for existing EEG-only clients.
  - System status/capabilities responses should enumerate Athena channels, sampling rates, and supported modalities for UI discovery.
- Processing pipeline alignment
  - EEG: reuse existing filters/feature extractors; ensure notch/bandpass defaults align with 14-bit noise floor.
  - fNIRS: blocked until BrainFlow exposes it; if not available, gate features and surface “not supported via BrainFlow” in API.
  - PPG: add HR/HRV extraction; use IMU for motion artifact rejection.
  - Fusion: define synchronized feature vector schema for ML models (EEG + PPG + IMU; add fNIRS later if/when available).

## Implementation plan
1) BrainFlow capability validation for Athena
   - Attempt discovery/stream using `BoardIds.MUSE_S_BLED_BOARD` via BLED112; confirm connection, channel count (EEG 4+1), sample rate, and whether PPG/IMU presets deliver data on Athena hardware.
   - If Athena fails to stream, open an upstream issue/PR to BrainFlow for an Athena board definition with BLE 5.3 characteristics.
2) BrainFlow-based driver
   - Replace current stub with BrainFlow-backed `MuseSAthenaDevice` using `BoardShim`.
   - Map presets:
     - DEFAULT_PRESET -> EEG; call `config_board("p50")` to enable AUX EEG channel if supported.
     - AUXILIARY_PRESET -> IMU; stream accel/gyro.
     - ANCILLARY_PRESET -> PPG; call `config_board("p61")` to enable.
   - Implement start/stop by calling `prepare_session`, `start_stream`, `stop_stream`, `release_session`; handle reconnection and exceptions.
3) Buffering and resampling
   - Pull data via `BoardShim.get_board_data()` or `get_board_data_count` + `get_current_board_data`.
   - Store per-modality ring buffers keyed by preset; resample EEG (likely 256 Hz) and align with PPG/IMU timestamps from BrainFlow.
4) Signal-quality and calibration
   - Surface BrainFlow-provided contact/ppg confidence if available; otherwise expose only availability flags and sample counts.
   - Provide baseline routines (eyes open/closed) for EEG; skip fNIRS until available.
5) API surface
   - Add device selection and capability flags noting “fNIRS not available via BrainFlow yet.”
   - Extend websocket payloads to include PPG/IMU when enabled; gate fNIRS fields.
6) Processing hooks
   - Wire EEG + PPG + IMU into feature extraction; add HR/HRV from PPG and motion artifact rejection using IMU.
7) Testing and validation
   - Unit tests mocking BrainFlow `BoardShim` to verify connect/stream/stop paths and buffering.
   - Live test with Athena hardware to confirm BrainFlow compatibility; measure sample rates, drop rates, latency.
8) Deployment and observability
   - Structured logging around BrainFlow errors (`BrainFlowError`), reconnect attempts, and per-modality throughput.
   - Ops checklist: BLED112 driver install, COM port configuration, udev/permissions (Linux), firmware version.

## Open questions / follow-ups
- Does Athena remain compatible with the Muse S BLED BrainFlow profile via BLED112, or is a new board definition required?
- If compatible, do we still get AUX EEG and PPG via config commands p50/p61? Are sampling rates identical to Muse S G2?
- Is fNIRS exposed at all over BLE? If not, we must exclude it from the BrainFlow path and keep it as a “future/SDK only” feature.
- Do Athena BLE 5.3 characteristics introduce MTU/throughput changes requiring updated BrainFlow buffer sizes?
- Battery life and heat when streaming continuous EEG + PPG via BrainFlow; is charging while streaming safe/supported?
