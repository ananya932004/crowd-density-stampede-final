document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('uploadForm');
  const imageInput = document.getElementById('imageInput');
  const capacityInput = document.getElementById('capacity');
  const countsDiv = document.getElementById('counts');
  const alertBox = document.getElementById('alertBox');
  const canvas = document.getElementById('annotatedCanvas');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!imageInput.files.length) {
      alert('Please choose an image first');
      return;
    }

    const file = imageInput.files[0];
    const capacity = parseInt(capacityInput.value || '50', 10);

    const fd = new FormData();
    fd.append('image', file);
    fd.append('capacity', capacity);

    const highAccuracy = document.getElementById('highAccuracy').checked;
    const useDensity = document.getElementById('useDensity').checked;
    const densityThreshold = parseInt(document.getElementById('densityThreshold').value || '30', 10);
    fd.append('high_accuracy', highAccuracy ? '1' : '0');
    if (highAccuracy) {
      fd.append('model_name', document.getElementById('modelName').value);
      fd.append('imgsz', document.getElementById('imgsz').value);
      fd.append('conf', document.getElementById('conf').value);
      fd.append('nms_iou', document.getElementById('nms').value);
      fd.append('tile_size', document.getElementById('tileSize').value);
      fd.append('overlap', document.getElementById('overlap').value);
      fd.append('tiling', '1');
      fd.append('use_density', useDensity ? '1' : '0');
      fd.append('density_threshold', densityThreshold.toString());
    }

    alertBox.textContent = 'Analyzing...';
    countsDiv.innerHTML = '';

    try {
      const res = await fetch('/upload', { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Upload failed');

      // Render counts and diagnosis
      countsDiv.innerHTML = '<h3>Detected objects</h3>';
      const ul = document.createElement('ul');
      for (const [k, v] of Object.entries(data.counts || {})) {
        const li = document.createElement('li');
        li.textContent = `${k}: ${v}`;
        ul.appendChild(li);
      }
      // Show detection vs density counts
  const detP = document.createElement('p');
  const yoloPeople = data.people_count_detection || (data.counts && (data.counts.person || data.counts.people)) || 0;
  detP.textContent = `people: ${yoloPeople}`;
      ul.appendChild(detP);
      if (data.density_count !== null && data.density_count !== undefined) {
        const denP = document.createElement('p');
        denP.textContent = `Density count: ${data.density_count.toFixed(1)}`;
        ul.appendChild(denP);
      }
      countsDiv.appendChild(ul);

      // Alert
      alertBox.textContent = data.message || '';
      alertBox.className = data.risk ? 'alert danger' : 'alert safe';

      // Annotated image
      if (data.annotated_image) {
        const img = new Image();
        img.onload = () => {
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0);
        };
        img.src = data.annotated_image;
      }

      // Density heatmap (if provided)
      const heatDiv = document.getElementById('heatmap');
      heatDiv.innerHTML = '';
      if (data.density_heatmap) {
        const himg = new Image();
        himg.onload = () => {
          // show heatmap next to annotated image
          himg.style.maxWidth = '100%';
        };
        himg.src = data.density_heatmap;
        heatDiv.appendChild(himg);
      } else {
        // density heatmap not provided — intentionally do not show density-related warnings
      }
    } catch (err) {
      console.error(err);
      alertBox.textContent = 'Error: ' + err.message;
      alertBox.className = 'alert danger';
    }
  });

  const highCheckbox = document.getElementById('highAccuracy');
  const adv = document.getElementById('advanced');
  highCheckbox.addEventListener('change', () => {
    adv.style.display = highCheckbox.checked ? 'block' : 'none';
  });
});
