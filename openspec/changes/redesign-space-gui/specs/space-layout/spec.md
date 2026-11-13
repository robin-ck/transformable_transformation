# Delta: space-layout

## ADDED Requirements

### Requirement: Tabs per stage
The Space SHALL group its interface into a **New scene** tab (upload, create, jobs list) and a **Scene** tab (scene picker, viewer, downloads and the scene tools), and SHALL open on the Scene tab when at least one finished scene exists, otherwise on New scene.

#### Scenario: First visit
- **WHEN** a user opens the Space and no finished scene exists
- **THEN** the New scene tab is shown

#### Scenario: Returning user
- **WHEN** a user opens the Space and finished scenes exist
- **THEN** the Scene tab is shown with the newest finished scene opened at its newest version

### Requirement: Tools next to the viewer
The Scene tab SHALL show the viewer and a tool panel side by side on desktop widths, with sub-tabs History, Add object and Preview, so the viewer stays visible while a tool is used. On narrow screens the panel MAY stack below the viewer.

#### Scenario: Switch tool
- **WHEN** the user switches from Add object to Preview
- **THEN** the viewer keeps its scene, version and camera

#### Scenario: Placement in progress
- **WHEN** the user switches sub-tab while an object is being placed
- **THEN** the placement is kept and resumes when they return to Add object

### Requirement: Finished-job notice
The Space SHALL notify the user when a reconstruction job finishes, and the notice SHALL let them open the new scene in the Scene tab.

#### Scenario: Reconstruction done
- **WHEN** the user is on the Scene tab and a reconstruction job finishes
- **THEN** a notice names the job and opens the new scene on request, without changing the current view by itself
