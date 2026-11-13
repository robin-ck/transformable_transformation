# Delta: scene-viewer

## ADDED Requirements

### Requirement: Interactive 3D view
The system SHALL display a splat scene in the browser with orbit, pan, zoom and a reset-camera control.

#### Scenario: Open scene
- **WHEN** the user opens a finished scene
- **THEN** it renders in the viewer and responds to mouse and touch navigation

### Requirement: Scene versions
The system SHALL store every edit as a new version, list versions with their prompt and date, and let the user switch between them without loss.

#### Scenario: Revert
- **WHEN** the user selects version 2 while viewing version 4
- **THEN** version 2 is displayed and versions 3 and 4 remain available

### Requirement: Download
The system SHALL let the user download the current version as `.ply` and as a compressed viewer file.

#### Scenario: Export
- **WHEN** the user clicks "Download .ply"
- **THEN** the file for the displayed version is downloaded
