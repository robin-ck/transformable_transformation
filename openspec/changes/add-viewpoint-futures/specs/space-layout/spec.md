# Delta: space-layout

## MODIFIED Requirements

### Requirement: Tools next to the viewer
The Scene tab SHALL show the viewer and a tool panel side by side on desktop widths, with sub-tabs History, Add object, Preview, Futures and Edit scene, so the viewer stays visible while a tool is used. On narrow screens the panel MAY stack below the viewer.

#### Scenario: Switch tool
- **WHEN** the user switches from Add object to Preview
- **THEN** the viewer keeps its scene, version and camera

#### Scenario: Placement in progress
- **WHEN** the user switches sub-tab while an object is being placed
- **THEN** the placement is kept and resumes when they return to Add object

#### Scenario: Region box only in Edit scene
- **WHEN** the user has drawn a region box in Edit scene and switches to another sub-tab
- **THEN** the box is removed, the viewer orbits again, and "Limit to a region" is switched off

#### Scenario: Open Futures
- **WHEN** the user opens the Futures sub-tab on a scene with viewpoints
- **THEN** the viewpoints are listed with thumbnails, the viewer keeps its camera until one is picked, and the gallery of the last used viewpoint is shown

#### Scenario: Scene without viewpoints
- **WHEN** the open scene has no viewpoints
- **THEN** the Futures sub-tab offers "Save current view" and explains that futures are made from saved views
