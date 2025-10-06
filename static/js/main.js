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

let progressTimer = null;
let progressValue = 0;
let progressStartedAt = null;
let progressHideTimeout = null;

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

function formatDuration(totalSeconds) {
  const safeSeconds = Math.max(0, Math.round(totalSeconds));
  const minutes = Math.floor(safeSeconds / 60);
  const seconds = safeSeconds % 60;
  const mm = String(minutes).padStart(2, "0");
  const ss = String(seconds).padStart(2, "0");
  return `${mm}:${ss}`;
}

function resetProgress(label = "Idle") {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
  if (progressHideTimeout) {
    clearTimeout(progressHideTimeout);
    progressHideTimeout = null;
  }
  progressStartedAt = null;
  progressValue = 0;
  progressFill.style.width = "0%";
  progressFill.classList.remove("success", "error");
  progressTrack.classList.remove("active");
  progressLabel.textContent = label;
  elapsedTimeText.textContent = "00:00";
  remainingTimeText.textContent = "--:--";
}

function startProgress() {
  resetProgress("Uploading & preparing video...");
  progressTrack.classList.add("active");
  progressValue = 5;
  progressFill.style.width = `${progressValue}%`;
  progressStartedAt = Date.now();
  updateTimeDisplays();

  progressTimer = setInterval(() => {
    if (progressValue >= 90) {
      clearInterval(progressTimer);
      progressTimer = null;
      return;
    }

    const increment = Math.random() * 6;
    progressValue = Math.min(90, progressValue + increment);
    progressFill.style.width = `${progressValue}%`;
    updateTimeDisplays();

    progressLabel.textContent = progressValue < 40
      ? "Crunching frames..."
      : progressValue < 75
      ? "Analyzing motion..."
      : "Refining output...";
  }, 450);
}

function finishProgress(success) {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
  if (progressHideTimeout) {
    clearTimeout(progressHideTimeout);
  }

  progressValue = 100;
  progressFill.style.width = "100%";
  progressFill.classList.toggle("success", success);
  progressFill.classList.toggle("error", !success);
  progressLabel.textContent = success
    ? "Processing complete."
    : "Processing failed. Try again.";
  updateTimeDisplays(success ? "success" : "error");
  progressStartedAt = null;

  progressHideTimeout = setTimeout(() => {
    resetProgress(success ? "Ready for the next run" : "Idle");
  }, success ? 2000 : 2500);
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

function updateTimeDisplays(mode = null) {
  if (!progressStartedAt) {
    elapsedTimeText.textContent = "00:00";
    remainingTimeText.textContent = "--:--";
    return;
  }

  const elapsedMs = Date.now() - progressStartedAt;
  const elapsedSeconds = Math.max(0, elapsedMs / 1000);
  elapsedTimeText.textContent = formatDuration(elapsedSeconds);

  if (mode === "success") {
    remainingTimeText.textContent = "00:00";
    return;
  }

  if (mode === "error") {
    remainingTimeText.textContent = "--:--";
    return;
  }

  if (progressValue <= 5) {
    remainingTimeText.textContent = "--:--";
    return;
  }

  const estimatedTotalSeconds = elapsedSeconds / (progressValue / 100);
  const remainingSeconds = Math.max(0, estimatedTotalSeconds - elapsedSeconds);
  remainingTimeText.textContent = Number.isFinite(remainingSeconds)
    ? formatDuration(remainingSeconds)
    : "--:--";
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
resetProgress();

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

  if (!videoInput.files || !videoInput.files.length) {
    setStatus("Please upload a video before processing.", "error");
    return;
  }

  const formData = new FormData(optimizerForm);
  formData.append("video", videoInput.files[0]);

  setStatus("Processing video — this may take a moment...", "info");
  startProgress();
  processButton.disabled = true;

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "An unexpected error occurred.");
    }

    const downloadUrl = data.downloadUrl || null;
    setStatus(data.message || "Processing complete.", "success", downloadUrl);
    finishProgress(true);
  } catch (error) {
    setStatus(error.message, "error");
    finishProgress(false);
  } finally {
    processButton.disabled = false;
  }
});
