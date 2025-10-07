const methodButtons = document.querySelectorAll(".method-button");
const methodInput = document.getElementById("method-input");
const parameterGroups = document.querySelectorAll(".parameter-group");
const optimizerForm = document.getElementById("optimizer-form");
const processButton = document.getElementById("process-button");
const statusArea = document.getElementById("status-area");
const videoInput = document.getElementById("video");
const dropzone = document.getElementById("dropzone");
const uploadLabel = document.getElementById("upload-label");
const progressTrack = document.getElementById("progress-track");
const progressFill = document.getElementById("progress-fill");
const progressLabel = document.getElementById("progress-label");
const elapsedTimeText = document.getElementById("elapsed-time");
const remainingTimeText = document.getElementById("remaining-time");
const profileOptions = document.querySelectorAll(".profile-option");
const dedupProfileInput = document.getElementById("dedup_profile_input");

let activeJob = null;
let pollIntervalId = null;
let progressResetTimeout = null;

function setActiveMethod(method) {
  methodButtons.forEach((button) => {
    const isActive = button.dataset.method === method;
    button.classList.toggle("active", isActive);
  });
  methodInput.value = method;

  parameterGroups.forEach((group) => {
    group.classList.toggle("hidden", group.dataset.for !== method);
  });
}

function setDedupProfile(profile) {
  if (!dedupProfileInput) {
    return;
  }
  dedupProfileInput.value = profile;
  profileOptions.forEach((option) => {
    const isActive = option.dataset.profile === profile;
    option.classList.toggle("active", isActive);
    option.setAttribute("aria-pressed", String(isActive));
  });
}

function formatDuration(totalSeconds) {
  if (!Number.isFinite(totalSeconds)) {
    return "00:00";
  }

  const safeSeconds = Math.max(0, Math.round(totalSeconds));
  const minutes = Math.floor(safeSeconds / 60);
  const seconds = safeSeconds % 60;
  const mm = String(minutes).padStart(2, "0");
  const ss = String(seconds).padStart(2, "0");
  return `${mm}:${ss}`;
}

function clearProgressReset() {
  if (progressResetTimeout) {
    clearTimeout(progressResetTimeout);
    progressResetTimeout = null;
  }
}

function resetProgressUI(label = "Idle") {
  if (pollIntervalId) {
    clearInterval(pollIntervalId);
    pollIntervalId = null;
  }
  activeJob = null;
  clearProgressReset();
  progressTrack.classList.remove("active");
  progressFill.style.width = "0%";
  progressFill.classList.remove("success", "error");
  progressLabel.textContent = label;
  elapsedTimeText.textContent = "00:00";
  remainingTimeText.textContent = "--:--";
}

function scheduleProgressReset(label) {
  clearProgressReset();
  progressResetTimeout = setTimeout(() => {
    resetProgressUI(label);
  }, 2400);
}

function applyProgressSnapshot(job) {
  if (!job) {
    return;
  }

  progressTrack.classList.add("active");
  progressFill.classList.remove("success", "error");

  const reportedTotal = Number(job.total_frames);
  const processed = Number(job.processed_frames) || 0;
  const hasTotal = Number.isFinite(reportedTotal) && reportedTotal > 0;
  const safeTotal = hasTotal ? reportedTotal : Math.max(processed, 1);

  const rawPercent = Number(job.progress);
  const percent = Number.isFinite(rawPercent)
    ? rawPercent
    : hasTotal
    ? (processed / safeTotal) * 100
    : 0;
  const clampedPercent = Math.min(100, Math.max(0, percent));
  progressFill.style.width = `${clampedPercent.toFixed(1)}%`;

  const stage = job.stage || "Processing";
  const framesLabel = hasTotal
    ? `${processed}/${safeTotal} frames`
    : `${processed} ${processed === 1 ? "frame" : "frames"}`;
  const percentLabel = `${clampedPercent.toFixed(1)}%`;
  const fpsValue = Number(job.fps);
  const labelParts = [stage, framesLabel];
  if (hasTotal) {
    labelParts.push(percentLabel);
  }
  if (Number.isFinite(fpsValue) && fpsValue > 0) {
    labelParts.push(`${fpsValue.toFixed(1)} fps`);
  }
  progressLabel.textContent = labelParts.join(" • ");

  const elapsed = Number(job.elapsed_seconds);
  elapsedTimeText.textContent = Number.isFinite(elapsed)
    ? formatDuration(elapsed)
    : "00:00";

  const eta = Number(job.eta_seconds);
  if (job.status === "completed") {
    remainingTimeText.textContent = "00:00";
  } else if (job.status === "failed") {
    remainingTimeText.textContent = "--:--";
  } else if (Number.isFinite(eta) && eta >= 0) {
    remainingTimeText.textContent = formatDuration(eta);
  } else {
    remainingTimeText.textContent = "--:--";
  }
}

function stopJobTracking() {
  if (pollIntervalId) {
    clearInterval(pollIntervalId);
    pollIntervalId = null;
  }
  activeJob = null;
}

async function pollJobProgress() {
  if (!activeJob) {
    return;
  }

  try {
    const response = await fetch(activeJob.progressUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Progress request failed with status ${response.status}`);
    }

    const job = await response.json();
    applyProgressSnapshot(job);

    if (job.status === "completed") {
      handleJobSuccess(job);
    } else if (job.status === "failed") {
      handleJobFailure(job);
    }
  } catch (error) {
    console.error(error);
    stopJobTracking();
    progressFill.classList.add("error");
    progressLabel.textContent = "Progress unavailable";
    remainingTimeText.textContent = "--:--";
    setStatus("Lost connection to progress updates. Please try again.", "error");
    processButton.disabled = false;
    scheduleProgressReset("Idle");
  }
}

function startJobTracking(jobId, progressUrl) {
  clearProgressReset();
  activeJob = { id: jobId, progressUrl };
  progressTrack.classList.add("active");
  progressFill.classList.remove("success", "error");
  progressFill.style.width = "0%";
  progressLabel.textContent = "Initializing...";
  elapsedTimeText.textContent = "00:00";
  remainingTimeText.textContent = "--:--";

  pollJobProgress();
  pollIntervalId = setInterval(pollJobProgress, 600);
}

function handleJobSuccess(job) {
  stopJobTracking();
  progressFill.classList.add("success");
  applyProgressSnapshot(job);
  const downloadUrl = job.download_url || (job.output_file ? `/download/${job.output_file}` : null);
  setStatus(job.message || "Video processed successfully.", "success", downloadUrl);
  processButton.disabled = false;
  scheduleProgressReset("Ready for the next run");
}

function handleJobFailure(job) {
  stopJobTracking();
  progressFill.classList.add("error");
  applyProgressSnapshot(job);
  const errorMessage = job.error || job.message || "Processing failed. Try again.";
  setStatus(errorMessage, "error");
  processButton.disabled = false;
  scheduleProgressReset("Idle");
}

function setStatus(message, type = "info", downloadUrl = null) {
  statusArea.innerHTML = "";
  const paragraph = document.createElement("p");
  paragraph.className = `status-message ${type}`.trim();
  paragraph.textContent = message;
  statusArea.appendChild(paragraph);

  if (downloadUrl) {
    const anchor = document.createElement("a");
    anchor.href = downloadUrl;
    anchor.className = "status-download";
    anchor.textContent = "Download processed video";
    anchor.target = "_blank";
    anchor.rel = "noopener";
    statusArea.appendChild(anchor);
  }
}

function updateUploadLabel() {
  if (videoInput.files && videoInput.files[0]) {
    uploadLabel.textContent = videoInput.files[0].name;
  } else {
    uploadLabel.textContent = "Drag & Drop or Click to Upload";
  }
}

methodButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setActiveMethod(button.dataset.method);
  });
});

setActiveMethod(methodInput.value);
resetProgressUI();

if (dedupProfileInput) {
  setDedupProfile(dedupProfileInput.value || "balanced");
}

profileOptions.forEach((option) => {
  option.addEventListener("click", () => {
    const profile = option.dataset.profile;
    if (!profile) {
      return;
    }
    setDedupProfile(profile);
  });
});

videoInput.addEventListener("change", updateUploadLabel);

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const items = event.dataTransfer?.files;
  if (items && items.length > 0) {
    videoInput.files = items;
    updateUploadLabel();
  }
});

optimizerForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (activeJob) {
    setStatus("A video is already being processed. Please wait for it to finish.", "info");
    return;
  }

  if (!videoInput.files || !videoInput.files.length) {
    setStatus("Please upload a video before processing.", "error");
    return;
  }

  const formData = new FormData(optimizerForm);
  formData.append("video", videoInput.files[0]);

  setStatus("Processing video — this may take a moment...", "info");
  processButton.disabled = true;
  stopJobTracking();
  clearProgressReset();
  progressTrack.classList.add("active");
  progressFill.classList.remove("success", "error");
  progressFill.style.width = "0%";
  progressLabel.textContent = "Uploading video...";
  elapsedTimeText.textContent = "00:00";
  remainingTimeText.textContent = "--:--";

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "An unexpected error occurred.");
    }
    const { jobId, progressUrl } = data;
    if (!jobId || !progressUrl) {
      throw new Error("Server response missing progress tracking data.");
    }

    setStatus(data.message || "Processing started.", "info");
    startJobTracking(jobId, progressUrl);
  } catch (error) {
    setStatus(error.message, "error");
    progressFill.classList.add("error");
    scheduleProgressReset("Idle");
    processButton.disabled = false;
  }
});
