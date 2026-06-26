/* ============================================
   torch-sla "Recommended Setup" interactive widget.
   Vanilla JS, no framework, no CDN — works offline on RTD.
   Fills #sla-recommend with OS / Device / Problem-size selectors
   and emits the recommended backend + exact pip install command.
   ============================================ */
(function () {
  "use strict";

  var STRUMPACK_URL = "https://github.com/sparsexlab/torch-strumpack/releases";
  var AMGX_URL = "https://github.com/sparsexlab/torch-amgx/releases";

  // Option definitions.
  var OS_OPTS = [
    { id: "linux", label: "Linux" },
    { id: "windows", label: "Windows" },
    { id: "macos", label: "macOS" }
  ];
  var DEVICE_OPTS = [
    { id: "cpu", label: "CPU" },
    { id: "cuda", label: "NVIDIA CUDA" },
    { id: "rocm", label: "AMD ROCm" }
  ];
  var SIZE_OPTS = [
    { id: "small", label: "< 2M DOF (direct)" },
    { id: "large", label: "> 2M DOF (iterative)" },
    { id: "multigpu", label: "Multi-GPU" }
  ];

  // Which devices each OS supports (macOS has no CUDA/ROCm).
  function deviceAvailable(os, device) {
    if (os === "macos") return device === "cpu";
    return true;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // Core recommendation logic. Returns { backend, cmd, note }.
  function recommend(os, device, size) {
    var pip = "pip install torch-sla";
    var releaseNote =
      "STRUMPACK and AmgX are <strong>not</strong> pip extras — install the " +
      'matching prebuilt wheel from <a href="' + STRUMPACK_URL + '">torch-strumpack ' +
      'Releases</a> / <a href="' + AMGX_URL + '">torch-amgx Releases</a> ' +
      "(ABI-tied to your CUDA + PyTorch version).";

    // Multi-GPU is handled first — distributed path regardless of vendor.
    if (size === "multigpu") {
      return {
        backend: "DSparseTensor (distributed) + pytorch CG",
        cmd: pip,
        note:
          "Multi-GPU / multi-node uses the distributed <code>DSparseTensor</code> " +
          "path with the device-agnostic PyTorch Krylov solvers — no extra wheel " +
          "needed. Works on CUDA and ROCm clusters."
      };
    }

    if (device === "cpu") {
      if (size === "large") {
        return {
          backend: "pytorch (CG / BiCGStab)",
          cmd: pip,
          note:
            "CPU <em>can</em> handle large and distributed problems via the " +
            "device-agnostic PyTorch Krylov solvers — it just won't be the " +
            "fastest. Add <code>[pyamg]</code> for an AMG preconditioner: " +
            "<code>pip install torch-sla[pyamg]</code>."
        };
      }
      // small / direct on CPU
      return {
        backend: "scipy (LU direct)",
        cmd: pip,
        note:
          "SciPy ships with the core install and gives a solid CPU direct LU. " +
          "For a portable multifrontal direct solver add STRUMPACK (prebuilt " +
          'wheel from <a href="' + STRUMPACK_URL + '">torch-strumpack Releases</a>).'
      };
    }

    if (device === "cuda") {
      if (size === "small") {
        return {
          backend: "cudss (Cholesky / LU direct, GPU)",
          cmd: pip + "[cudss]",
          note:
            "cuDSS is the fastest path for medium GPU problems (~10K-2M DOF), " +
            "NVIDIA-only. For AMG-friendly very large systems, also grab the " +
            'AmgX wheel from <a href="' + AMGX_URL + '">torch-amgx Releases</a>.'
        };
      }
      // large / iterative on CUDA
      return {
        backend: "pytorch (CG / BiCGStab) — or AmgX for AMG-friendly systems",
        cmd: pip + "[cudss]",
        note:
          "For > 2M DOF the device-agnostic PyTorch Krylov solvers scale on " +
          "GPU. If your system is AMG-friendly, the GPU AMG/Krylov AmgX backend " +
          'is often fastest — install its wheel from <a href="' + AMGX_URL +
          '">torch-amgx Releases</a> (not a pip extra). cuDSS stays useful for ' +
          "direct sub-solves."
      };
    }

    if (device === "rocm") {
      if (size === "small") {
        return {
          backend: "strumpack (multifrontal LU direct, ROCm)",
          cmd: pip + "[pyamg]",
          note:
            "cuDSS is NVIDIA-only, so on AMD ROCm the direct path is STRUMPACK " +
            "(GPU multifrontal LU). " + releaseNote +
            " <code>[pyamg]</code> adds an on-device V-cycle preconditioner."
        };
      }
      // large / iterative on ROCm
      return {
        backend: "pytorch (CG / BiCGStab) — or strumpack direct",
        cmd: pip + "[pyamg]",
        note:
          "The PyTorch-native Krylov solvers are device-agnostic and run on " +
          "ROCm. For a direct solve use STRUMPACK's ROCm wheel (" +
          '<a href="' + STRUMPACK_URL + '">Releases</a>). cuDSS / AmgX are ' +
          "NVIDIA-only and do not apply here."
      };
    }

    return null;
  }

  function render(root) {
    var state = { os: "linux", device: "cuda", size: "small" };

    root.innerHTML = "";

    var title = document.createElement("p");
    title.className = "sla-rec-title";
    title.textContent = "Pick your environment → get the recommended backend + install command";
    root.appendChild(title);

    var controls = document.createElement("div");
    controls.className = "sla-rec-controls";
    root.appendChild(controls);

    var result = document.createElement("div");
    result.className = "sla-rec-result";
    root.appendChild(result);

    function makeField(labelText, opts, key) {
      var field = document.createElement("div");
      field.className = "sla-rec-field";

      var label = document.createElement("label");
      label.textContent = labelText;
      field.appendChild(label);

      var btnWrap = document.createElement("div");
      btnWrap.className = "sla-rec-btns";
      field.appendChild(btnWrap);

      opts.forEach(function (opt) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "sla-rec-opt";
        b.textContent = opt.label;
        b.setAttribute("data-key", key);
        b.setAttribute("data-val", opt.id);
        b.addEventListener("click", function () {
          if (b.disabled) return;
          state[key] = opt.id;
          // If device became unavailable for the chosen OS, fall back to CPU.
          if (key === "os" && !deviceAvailable(state.os, state.device)) {
            state.device = "cpu";
          }
          update();
        });
        btnWrap.appendChild(b);
      });

      controls.appendChild(field);
      return btnWrap;
    }

    makeField("Operating system", OS_OPTS, "os");
    makeField("Device", DEVICE_OPTS, "device");
    makeField("Problem size", SIZE_OPTS, "size");

    function update() {
      // Sync pressed/disabled state on every button.
      var btns = controls.querySelectorAll("button.sla-rec-opt");
      Array.prototype.forEach.call(btns, function (b) {
        var key = b.getAttribute("data-key");
        var val = b.getAttribute("data-val");
        var disabled = key === "device" && !deviceAvailable(state.os, val);
        b.disabled = disabled;
        b.setAttribute("aria-pressed", String(!disabled && state[key] === val));
      });

      var rec = recommend(state.os, state.device, state.size);
      result.innerHTML = "";
      if (!rec) return;

      var backend = document.createElement("p");
      backend.className = "sla-rec-backend";
      backend.innerHTML =
        "Recommended backend: <strong>" + escapeHtml(rec.backend) + "</strong>";
      result.appendChild(backend);

      var cmdWrap = document.createElement("div");
      cmdWrap.className = "sla-rec-cmd-wrap";

      var pre = document.createElement("pre");
      pre.className = "sla-rec-cmd";
      pre.textContent = rec.cmd;
      cmdWrap.appendChild(pre);

      var copy = document.createElement("button");
      copy.type = "button";
      copy.className = "sla-rec-copy";
      copy.textContent = "Copy";
      copy.addEventListener("click", function () {
        var done = function () {
          copy.textContent = "Copied!";
          setTimeout(function () { copy.textContent = "Copy"; }, 1500);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(rec.cmd).then(done, done);
        } else {
          var ta = document.createElement("textarea");
          ta.value = rec.cmd;
          document.body.appendChild(ta);
          ta.select();
          try { document.execCommand("copy"); } catch (e) {}
          document.body.removeChild(ta);
          done();
        }
      });
      cmdWrap.appendChild(copy);
      result.appendChild(cmdWrap);

      if (rec.note) {
        var note = document.createElement("p");
        note.className = "sla-rec-note";
        note.innerHTML = rec.note;
        result.appendChild(note);
      }
    }

    update();
  }

  function init() {
    var root = document.getElementById("sla-recommend");
    if (root) render(root);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
