# Delta: text-editing

## ADDED Requirements

### Requirement: Add object from text
The system SHALL generate a 3D object from a text prompt, show it in the viewer, let the user move, rotate and scale it, and on confirmation merge it into a new scene version.

#### Scenario: Add a tree
- **WHEN** the user types "a mature oak tree" and confirms placement
- **THEN** a new version contains the original scene plus the tree at the chosen position and scale

#### Scenario: Cancel placement
- **WHEN** the user cancels before confirming
- **THEN** no new version is created

### Requirement: Edit scene from text
The system SHALL apply a text instruction to the existing scene (e.g. "make the façade green") as a remote job and SHALL save the result as a new version.

#### Scenario: Successful edit
- **WHEN** the user submits an edit instruction on version 1
- **THEN** a job runs and, when done, version 2 shows the edited scene

#### Scenario: Optional region
- **WHEN** the user marks a region in the view before submitting
- **THEN** the edit is limited to that region

### Requirement: Text-to-image preview
The system SHALL render the current camera view and restyle it from a text prompt into a 2D image, without changing the 3D scene.

#### Scenario: Concept preview
- **WHEN** the user types "same street in autumn with rain gardens"
- **THEN** a 2D image is shown next to the viewer and can be downloaded, and the scene version is unchanged

### Requirement: Prompt log
The system SHALL record every prompt, mode, model name and resulting version, so edits are traceable.

#### Scenario: Inspect history
- **WHEN** the user opens the history of version 3
- **THEN** they see the prompt and model that produced it
