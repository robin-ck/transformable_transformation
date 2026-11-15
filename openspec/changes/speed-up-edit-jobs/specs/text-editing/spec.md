# Delta: text-editing

## ADDED Requirements

### Requirement: Job turnaround
The system SHALL finish edit jobs fast enough to try out ideas one after another. On the test scene (1.55 M Gaussians), counting from the start of the job's GPU container and leaving out time spent waiting for a free GPU:
- an Edit scene job SHALL finish within 4 minutes and SHALL cost at most $0.15 (estimate in the run log)
- a Preview SHALL finish within 90 seconds
- a Preview started within 5 minutes after the previous one finished SHALL finish within 30 seconds
- an Add object job started within 5 minutes after the previous one finished SHALL finish within 60 seconds

Faster settings SHALL NOT visibly weaken the reference edit.

#### Scenario: Reference edit
- **WHEN** the user runs "make the wooden table bright green" with a region around the table on the test scene
- **THEN** the job finishes within 4 minutes, and the new version shows the table clearly green and the rest of the room unchanged

#### Scenario: Previews back to back
- **WHEN** the user starts a second preview within 5 minutes after the first one finished
- **THEN** the second preview is shown within 30 seconds of its job starting

#### Scenario: First preview of a session
- **WHEN** no preview has run in the last 5 minutes
- **THEN** the preview is shown within 90 seconds of its GPU container starting

#### Scenario: Edit on a freshly edited version
- **WHEN** the user edits the version that the previous edit job produced
- **THEN** the job reuses that version's scene file and does not download it from the dataset again
