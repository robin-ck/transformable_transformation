# Delta: scene-viewer

## MODIFIED Requirements

### Requirement: Scene versions
The system SHALL store every edit as a new version, list versions with their prompt and date, and let the user switch between them without loss. The list SHALL show, for each version, its number, date, mode (reconstruction, add object, edit scene), prompt, model and the version it was made from, newest first. Previews and edits that have not produced a version (running or failed) SHALL appear in the same list, marked as such, and SHALL NOT open a version.

#### Scenario: Revert
- **WHEN** the user selects version 2 while viewing version 4
- **THEN** version 2 is displayed and versions 3 and 4 remain available

#### Scenario: Read history
- **WHEN** the user opens the History sub-tab for a scene with versions 1 to 3, where v3 added "a mature oak tree" to v2
- **THEN** the v3 row shows the date, "add object", the prompt "a mature oak tree", the model names and "from v2"

## ADDED Requirements

### Requirement: Hide objects
For a version made with Add object, the system SHALL list the objects added up to that version and let the user show or hide each one in the viewer. Hiding SHALL change only the view: no version is created and downloads keep all objects.

#### Scenario: Hide a tree
- **WHEN** the user hides "a mature oak tree" on v3
- **THEN** the viewer shows v3 without the tree, and "Download .ply" still gives the full v3

#### Scenario: Older version
- **WHEN** a version was made before object tracking existed
- **THEN** no hide controls are shown for it, and the version still opens normally
