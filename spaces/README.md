---
title: ORBIT Dashboard Demo
emoji: "\U0001F916"
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.41.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Robot learning failure analysis dashboard
---

# ORBIT Dashboard Demo

Interactive dashboard for analyzing robot learning deployment failures.
This demo uses synthetic deployment data (20 episodes: 12 success, 5 lighting failures, 3 position failures).

**Features:**
- Session Overview with success rate metrics and episode timeline
- Failure Analysis with heuristic detection (stall, gripper drop, reward, timeout, bounds)
- Prescriptions with priority-ranked corrective recommendations
- Export to JSON, CSV, or Markdown

[Full repository on GitHub](https://github.com/Rahillasne/Orbit)
