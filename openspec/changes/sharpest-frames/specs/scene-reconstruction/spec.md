# Delta: scene-reconstruction

## ADDED Requirements

### Requirement: Sharp video frames
When the input is a video, the system SHALL take more frames than needed and keep the sharpest frame of each equal stretch of the video, so that frames blurred by camera motion are left out while the kept frames still cover the whole video in time order.

#### Scenario: Blurred frames are skipped
- **WHEN** a video has stretches where most frames are blurred by a quick camera move but some are sharp
- **THEN** the frames used for pose estimation and training are the sharp ones from each stretch

#### Scenario: Frame count unchanged
- **WHEN** the user creates a scene from a video with the Standard preset
- **THEN** about 180 frames are used, spread over the whole video, as before this change
