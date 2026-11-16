# Delta: text-editing

## MODIFIED Requirements

### Requirement: Text-to-image preview
The system SHALL render the current camera view and restyle it from a text prompt into a 2D image, without changing the 3D scene. Every preview SHALL be kept with its prompt and source version, and the user SHALL be able to see the scene's earlier previews and open any finished one, also after reloading or closing the page. When the preferred GPU type is not free, the preview SHALL run on another suitable GPU instead of waiting for it.

#### Scenario: Concept preview
- **WHEN** the user types "same street in autumn with rain gardens"
- **THEN** a 2D image is shown next to the viewer and can be downloaded, and the scene version is unchanged

#### Scenario: Preview finished after the page was closed
- **WHEN** a preview of v3 finishes while no page shows it, and the user later opens the scene
- **THEN** the Preview sub-tab lists it as done with its prompt and "v3", and clicking it shows the restyled image and the plain render

#### Scenario: Preview still running after a reload
- **WHEN** the user reloads the page while a preview is waiting for a GPU and clicks it in the list
- **THEN** its status line shows that it is waiting or running, and its images appear when it is done

#### Scenario: Failed preview
- **WHEN** the user clicks a failed preview in the list
- **THEN** its error message is shown and no image is shown

#### Scenario: Status in the prompt log
- **WHEN** a preview finished after the page that started it was closed
- **THEN** the prompt log shows it as done, not queued

#### Scenario: No L4 free
- **WHEN** no L4 is free when a preview starts
- **THEN** the preview runs on an A10G or L40S, and its run log names that GPU
