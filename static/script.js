function updateFileName() {
  var input = document.getElementById('fileInput');
  var fileNameDisplay = document.getElementById('fileNameDisplay');
  fileNameDisplay.textContent = input.files[0].name;
}

async function removeFile(filename) {
      window.location.href = '/'
}


