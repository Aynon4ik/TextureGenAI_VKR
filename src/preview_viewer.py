from __future__ import annotations

from pathlib import Path
from urllib.parse import quote, urlencode

from src.preview_env import HDR_EXR, HDR_FROM_RU, hdr_env_status

PBR_VIEWER_HTML = (
    Path(__file__).resolve().parent.parent / "assets" / "preview" / "pbr_viewer.html"
)

IFRAME_TITLE = "PBR Preview"

_VIEWER_INNER_STYLE = (
    "width:100%;height:100%;min-height:100%;"
    "background:#f1f5f9;box-sizing:border-box;"
)

HDR_SWITCH_JS = "(label) => { window.pbrViewer && window.pbrViewer.setEnv(label); }"

PBR_VIEWER_HEAD_JS = """
window.pbrViewer = (function () {
  function findFrame(root) {
    var direct = root.getElementById && root.getElementById("pbr-viewer-frame");
    if (direct) return direct;
    var wrap = root.getElementById && root.getElementById("preview-viewer-wrap");
    if (wrap) {
      var inWrap = wrap.querySelector && wrap.querySelector("iframe");
      if (inWrap) return inWrap;
      if (wrap.shadowRoot) {
        var s = findFrame(wrap.shadowRoot);
        if (s) return s;
      }
    }
    var q = root.querySelector && root.querySelector("#pbr-viewer-frame, #preview-viewer-wrap iframe");
    if (q) return q;
    var all = root.querySelectorAll ? root.querySelectorAll("*") : [];
    for (var i = 0; i < all.length; i++) {
      if (all[i].shadowRoot) {
        var f = findFrame(all[i].shadowRoot);
        if (f) return f;
      }
    }
    return null;
  }
  function fileUrl(raw) {
    if (!raw) return null;
    if (typeof raw === "string") {
      if (raw.indexOf("/gradio_api/file=") === 0 || raw.indexOf("/file=") === 0) return raw;
      return "/gradio_api/file=" + encodeURIComponent(raw.replace(/\\\\/g, "/"));
    }
    if (raw.path) return fileUrl(raw.path);
    if (raw.url) return fileUrl(raw.url);
    return null;
  }
  function post(msg) {
    var frame = findFrame(document);
    if (frame && frame.contentWindow) frame.contentWindow.postMessage(msg, "*");
  }
  return {
    loadModel: function (raw) {
      var url = fileUrl(raw);
      if (!url) return;
      var msg = { type: "loadModel", url: url };
      var n = 0;
      (function tick() {
        post(msg);
        if (++n < 40) setTimeout(tick, 300);
      })();
    },
    setEnv: function (label) {
      post({ type: "setEnv", label: label });
    }
  };
})();
"""


def viewer_placeholder_html() -> str:
    return (
        f'<div id="pbr-viewer-placeholder" style="'
        f"{_VIEWER_INNER_STYLE}"
        f"display:flex;align-items:center;justify-content:center;color:#64748b;"
        f"font:14px/1.5 system-ui,sans-serif;text-align:center;padding:24px;"
        f'">'
        f"Нажмите «Показать на модели»</div>"
    )


def viewer_loading_html(message: str, percent: int) -> str:
    pct = max(0, min(100, int(percent)))
    return (
        f'<div class="preview-viewer-loading" style="'
        f"{_VIEWER_INNER_STYLE}"
        f"display:flex;flex-direction:column;align-items:center;justify-content:center;"
        f'gap:12px;padding:24px;">'
        f'<div style="color:#334155;font:14px/1.4 system-ui,sans-serif;text-align:center;">'
        f"{message} — {pct}%</div>"
        f'<div style="width:72%;max-width:420px;height:6px;background:#e2e8f0;'
        f'border-radius:4px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,#667eea,#764ba2);'
        f'border-radius:4px;transition:width 0.15s ease;"></div>'
        f"</div></div>"
    )


def gradio_file_url(path: Path | str) -> str:
    p = str(Path(path).resolve()).replace("\\", "/")
    return f"/gradio_api/file={quote(p, safe='/')}"


def viewer_iframe_html(
    hdr_label: str = "Студия",
    glb_url: str | None = None,
) -> str:
    env_key = HDR_FROM_RU.get(hdr_label, "studio")
    err = hdr_env_status()
    if err:
        return f"<p style='color:#666;padding:12px'>{err}</p>"
    params: dict[str, str] = {
        "studio": gradio_file_url(HDR_EXR["studio"]),
        "outdoor": gradio_file_url(HDR_EXR["outdoor"]),
        "env": env_key,
    }
    if glb_url:
        params["glb"] = glb_url
    q = urlencode(params)
    src = f"{gradio_file_url(PBR_VIEWER_HTML)}?{q}"
    return (
        f'<iframe id="pbr-viewer-frame" title="{IFRAME_TITLE}" '
        f'src="{src}" allow="fullscreen" allowfullscreen '
        f'style="{_VIEWER_INNER_STYLE}border:none;display:block;"></iframe>'
    )
