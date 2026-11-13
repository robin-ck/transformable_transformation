# Delta: scene-reconstruction

## ADDED Requirements

### Requirement: Upload input media
The system SHALL accept either one video (MP4/MOV, ≤ 2 min, ≤ 500 MB) or 20 to 300 photos (JPG/PNG) and SHALL reject other inputs with a clear message.

#### Scenario: Valid video
- **WHEN** the user uploads a 45 s MP4
- **THEN** the system stores it and offers a "Create scene" button

#### Scenario: Too few photos
- **WHEN** the user uploads 8 photos
- **THEN** the system rejects the upload and states that at least 20 photos are needed

### Requirement: Remote reconstruction job
The system SHALL run reconstruction (frame extraction, camera pose estimation, Gaussian splat training) as a job on a remote GPU and SHALL return a `.ply` scene plus a viewer-ready compressed file.

#### Scenario: Successful reconstruction
- **WHEN** the user clicks "Create scene" on a valid upload
- **THEN** a job is submitted, its status is shown, and on completion the scene opens in the viewer as version 1

#### Scenario: Pose estimation fails
- **WHEN** camera poses cannot be recovered (e.g. too little overlap)
- **THEN** the job is marked failed with a human-readable reason and a tip on how to film better

### Requirement: Job status
The system SHALL show each job's state (queued, running, done, failed) and elapsed time, and SHALL keep the job result available after the user closes and reopens the page.

#### Scenario: User returns later
- **WHEN** the user reopens the Space after a job finished
- **THEN** the finished scene is listed and can be opened
