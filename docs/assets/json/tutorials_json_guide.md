# MEYELens Tutorials JSON Guide

This guide explains how to add tutorials to the MEYELens website using a JSON-based structure.

The idea is simple:

- `tutorials.json` stores the tutorial metadata and page content.
- Each tutorial can point to an associated Python file.
- A single tutorial template page renders all tutorials automatically.
- Tutorial cards in the Resources page are generated from the same JSON file.

Recommended files:

```text
assets/json/tutorials.json
assets/js/tutorials.js
tutorial-detail.html
assets/tutorials/tutorial_preview.py
assets/tutorials/tutorial_online_recording.py
assets/tutorials/tutorial_offline_recording.py
```

---

## 1. Basic JSON structure

`tutorials.json` should contain an array of tutorial objects:

```json
[
  {
    "id": "first-preview",
    "title": "First camera preview",
    "tag": "Basic",
    "level": "Beginner",
    "summary": "Initialize a camera and open a live preview using the Python API.",
    "description": "This tutorial checks that MEYELens is installed correctly and that your camera can be opened from Python.",
    "image": "assets/media/tutorial_preview_crop.gif",
    "imageAlt": "First camera preview tutorial",
    "python_file": "assets/tutorials/tutorial_preview.py",
    "button": "Open tutorial",
    "sections": []
  }
]
```

---

## 2. Core fields

These fields define the tutorial card and the tutorial detail page.

| Field | Required | Description |
|---|---:|---|
| `id` | Yes | Unique tutorial identifier. Used in URLs such as `tutorial-detail.html?id=first-preview`. |
| `title` | Yes | Tutorial title shown in the card and detail page. |
| `tag` | Recommended | Short category label, for example `Basic`, `Online recording`, `Analysis`. |
| `level` | Optional | Difficulty level, for example `Beginner`, `Intermediate`, `Advanced`. |
| `summary` | Recommended | Short text used in tutorial cards. |
| `description` | Recommended | Longer description used at the top of the tutorial detail page. |
| `image` | Optional | Preview image shown in the tutorial card and/or detail page. |
| `imageAlt` | Optional | Alternative text for the preview image. |
| `python_file` | Optional | Path to the Python script associated with the tutorial. |
| `button` | Optional | Button label for the tutorial card. Default can be `Open tutorial`. |
| `sections` | Recommended | Ordered list of content sections rendered in the tutorial page. |

Example:

```json
{
  "id": "online-recording",
  "title": "Record online pupil predictions",
  "tag": "Online recording",
  "level": "Intermediate",
  "summary": "Run live pupil prediction, save CSV files, and send keyboard trigger pulses.",
  "description": "This tutorial shows how to record online pupil predictions with synchronized keyboard triggers.",
  "image": "assets/media/tutorial_online_recording.png",
  "imageAlt": "Online pupil recording tutorial preview",
  "python_file": "assets/tutorials/tutorial_online_recording.py",
  "button": "Open tutorial",
  "sections": []
}
```

---

## 3. Sections

The `sections` field is an ordered array. Each section has a `kind` field that controls how it is rendered.

General structure:

```json
"sections": [
  {
    "kind": "text",
    "title": "Goal",
    "body": "Open a live camera preview and verify that the camera works."
  }
]
```

Supported `kind` values:

```text
text
note
tip
warning
list
image
code
link
```

---

## 4. Text section

Use `kind: "text"` for normal tutorial paragraphs.

```json
{
  "kind": "text",
  "title": "Goal",
  "body": "Open a live camera preview and verify that the camera and MEYELens model can run from Python."
}
```

Recommended rendering:

```html
<div class="text-block">
  <h2>Goal</h2>
  <p>...</p>
</div>
```

---

## 5. Tip section

Use `kind: "tip"` for helpful suggestions.

```json
{
  "kind": "tip",
  "title": "Camera index",
  "body": "If the wrong camera opens, try changing `camera_index=0` to `camera_index=1` or `camera_index=2`."
}
```

Recommended rendering:

```html
<div class="note note-tip">
  <strong>Camera index:</strong>
  ...
</div>
```

---

## 6. Warning section

Use `kind: "warning"` for important cautions.

```json
{
  "kind": "warning",
  "title": "Camera already in use",
  "body": "If the camera does not open, close other applications that may be using it."
}
```

Recommended rendering:

```html
<div class="note note-warning">
  <strong>Camera already in use:</strong>
  ...
</div>
```

---

## 7. Note section

Use `kind: "note"` for neutral information that is not specifically a tip or warning.

```json
{
  "kind": "note",
  "title": "Model configuration",
  "body": "This tutorial uses the default MEYELens model and configuration file."
}
```

Recommended rendering:

```html
<div class="note">
  <strong>Model configuration:</strong>
  ...
</div>
```

---

## 8. List section

Use `kind: "list"` for requirements, outputs, steps, or checklist-like content.

```json
{
  "kind": "list",
  "title": "Requirements",
  "items": [
    "MEYELens installed in a dedicated Python environment",
    "One compatible USB camera",
    "A visible eye region in the camera frame"
  ]
}
```

Recommended rendering:

```html
<div class="content-block">
  <h2>Requirements</h2>
  <ul class="list">
    <li>...</li>
  </ul>
</div>
```

---

## 9. Image section

Use `kind: "image"` to insert an image inside the tutorial.

```json
{
  "kind": "image",
  "title": "Example camera view",
  "src": "assets/media/tutorial_preview_crop.gif",
  "alt": "Example MEYELens camera preview",
  "caption": "Example preview of the eye-facing camera stream."
}
```

Recommended rendering:

```html
<figure class="media-figure">
  <img src="..." alt="...">
  <figcaption>...</figcaption>
</figure>
```

---

## 10. Code section

Use `kind: "code"` to display the associated Python file.

The simplest form uses the tutorial-level `python_file` field:

```json
{
  "kind": "code",
  "title": "Preview script",
  "source": "python_file"
}
```

This means: load the file defined here:

```json
"python_file": "assets/tutorials/tutorial_preview.py"
```

You can also point directly to a different code file:

```json
{
  "kind": "code",
  "title": "Alternative script",
  "source": "assets/tutorials/alternative_preview.py"
}
```

Recommended rendering:

```html
<div class="code-block">
  <div class="code-header">
    <span class="code-label">Python</span>
    <button class="copy-btn" type="button" aria-label="Copy code">Copy</button>
  </div>
  <pre><code class="language-python">...</code></pre>
</div>
```

---

## 11. Link section

Use `kind: "link"` for a small call-to-action block.

```json
{
  "kind": "link",
  "title": "Next tutorial",
  "body": "Continue with online recording after checking the camera preview.",
  "url": "tutorial-detail.html?id=online-recording",
  "button": "Open next tutorial"
}
```

Recommended rendering:

```html
<div class="content-block">
  <h2>Next tutorial</h2>
  <p>...</p>
  <a class="btn" href="...">Open next tutorial</a>
</div>
```

---

## 12. Lightweight Markdown support

Text fields such as `body`, `items`, and `caption` can support a small subset of Markdown.

Recommended supported syntax:

| Markdown | Output |
|---|---|
| `` `code` `` | Inline code |
| `**bold**` | Bold text |
| `[text](https://example.com)` | External link |

Example:

```json
{
  "kind": "tip",
  "title": "Installation",
  "body": "Install MEYELens from the [software page](software.html#installation) before running this tutorial. Use `camera_index=1` if needed."
}
```

Important: keep Markdown support simple. Do not put long HTML blocks in JSON unless you fully trust and control the file.

---

## 13. Complete example

```json
[
  {
    "id": "first-preview",
    "title": "First camera preview",
    "tag": "Basic",
    "level": "Beginner",
    "summary": "Initialize a camera and open a live preview using the Python API.",
    "description": "This tutorial checks that MEYELens is installed correctly and that your camera can be opened from Python.",
    "image": "assets/media/tutorial_preview_crop.gif",
    "imageAlt": "First camera preview tutorial",
    "python_file": "assets/tutorials/tutorial_preview.py",
    "button": "Open tutorial",

    "sections": [
      {
        "kind": "text",
        "title": "Goal",
        "body": "Open a live camera preview and verify that the camera and MEYELens model can run from Python."
      },
      {
        "kind": "list",
        "title": "Requirements",
        "items": [
          "MEYELens installed in a dedicated Python environment",
          "One compatible USB camera",
          "A visible eye region in the camera frame"
        ]
      },
      {
        "kind": "tip",
        "title": "Camera index",
        "body": "If the wrong camera opens, try changing `camera_index=0` to `camera_index=1` or `camera_index=2`."
      },
      {
        "kind": "code",
        "title": "Preview script",
        "source": "python_file"
      },
      {
        "kind": "warning",
        "title": "Camera already in use",
        "body": "If the camera does not open, close other applications that may be using it."
      },
      {
        "kind": "text",
        "title": "Expected result",
        "body": "A live preview window should open. You can then tune thresholds and check the quality of the pupil and eye masks."
      }
    ]
  }
]
```

---

## 14. Adding a new tutorial

To add a new tutorial:

1. Add a Python script in `assets/tutorials/`.
2. Add a new object to `assets/json/tutorials.json`.
3. Give it a unique `id`.
4. Set `python_file` to the path of the Python script.
5. Add ordered `sections` describing the tutorial.
6. Open the tutorial using:

```text
tutorial-detail.html?id=your-tutorial-id
```

Example:

```json
{
  "id": "offline-recording",
  "title": "Record video with triggers",
  "tag": "Offline recording",
  "level": "Beginner",
  "summary": "Save a camera stream to video while recording synchronized keyboard triggers to CSV.",
  "description": "This tutorial shows how to record raw video and trigger information for offline analysis.",
  "image": "assets/media/tutorial_offline_recording.png",
  "imageAlt": "Offline recording tutorial preview",
  "python_file": "assets/tutorials/tutorial_online_recording.py",
  "button": "Open tutorial",
  "sections": [
    {
      "kind": "text",
      "title": "Goal",
      "body": "Record the camera stream without running online segmentation."
    },
    {
      "kind": "code",
      "title": "Recording script",
      "source": "python_file"
    }
  ]
}
```

---

## 15. Suggested tutorial categories

Suggested values for `tag`:

```text
Basic
Online recording
Offline recording
Analysis
Gaze tracking
Pupillometry
Experiment
```

Suggested values for `level`:

```text
Beginner
Intermediate
Advanced
```

