# Delta: scene-reconstruction

## MODIFIED Requirements

### Requirement: Remote reconstruction job
The system SHALL run reconstruction (frame extraction, camera pose estimation, Gaussian splat training) as a job on a remote GPU and SHALL return a `.ply` scene plus a viewer-ready compressed file. The user SHALL be able to choose the training steps, the number of frames taken from a video and the maximum image size, through a preset (Fast, Standard, High) or directly, and SHALL see the estimated GPU time and cost before starting. Settings whose estimated time exceeds the per-job limit SHALL be refused before any GPU work. Each job SHALL record the settings it ran with.

#### Scenario: Successful reconstruction
- **WHEN** the user clicks "Create scene" on a valid upload
- **THEN** a job is submitted, its status is shown, and on completion the scene opens in the viewer as version 1

#### Scenario: Pose estimation fails
- **WHEN** camera poses cannot be recovered (e.g. too little overlap)
- **THEN** the job is marked failed with a human-readable reason and a tip on how to film better

#### Scenario: Default settings
- **WHEN** the user creates a scene without changing the settings
- **THEN** it is trained with the Standard preset (7,000 steps, about 180 frames, 1,600 px), as before this change

#### Scenario: Faster test run
- **WHEN** the user picks Fast for a video
- **THEN** the estimate shows a shorter time and lower cost than Standard, the job runs with Fast's settings, and the Jobs table shows them

#### Scenario: Custom settings
- **WHEN** the user moves the training steps slider away from every preset
- **THEN** the preset shows Custom and the estimate updates

#### Scenario: Too long
- **WHEN** the chosen settings are estimated to take longer than 45 min
- **THEN** the estimate says so, "Create scene" refuses to start with a message to lower the steps or frames, and no input is uploaded

#### Scenario: Large photos
- **WHEN** the user uploads photos whose long side is 4,000 px with a max image size of 1,600 px
- **THEN** the photos are scaled to 1,600 px on the long side before pose estimation and training
