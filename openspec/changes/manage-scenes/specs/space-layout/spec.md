# Delta: space-layout

## ADDED Requirements

### Requirement: Scene details
The Scene tab SHALL let the user see a scene's details, give it a name and notes, and delete it with everything made for it after a confirmation.

#### Scenario: Rename
- **WHEN** the user enters "Main street, before" as the name and saves
- **THEN** the scene list shows "Main street, before · <job id>", also after a reload

#### Scenario: Delete
- **WHEN** the user clicks Delete scene and confirms
- **THEN** the scene, its versions, edits, previews and generated objects are removed from the dataset, and the newest remaining scene opens

#### Scenario: Delete while a job runs
- **WHEN** an edit, preview or object job of the scene is still queued or running
- **THEN** deletion is refused with a message naming that job, and nothing is removed
