document.addEventListener("DOMContentLoaded", function () {
    const image = document.querySelector(".img-fluid");
    const imageDimensions = document.getElementById("image-dimensions");
    const rgbTableBody = document.querySelector("#rgb-table tbody");
    const showRgbTableButton = document.getElementById("showRgbTableButton");
    const rgbTable = document.getElementById("rgb-table");
 
    // Wait for the image to load
    image.addEventListener("load", function () {
      const width = this.width;
      const height = this.height;
      imageDimensions.innerText = `Image Dimensions: ${width} x ${height}`;
    });
 
    showRgbTableButton.addEventListener("click", function () {
      // Toggle the visibility of the RGB table
      rgbTable.classList.toggle("d-none");
 
      if (!rgbTable.classList.contains("d-none")) {
        // Get RGB values from the image when the table is visible
        const canvas = document.createElement("canvas");
        const context = canvas.getContext("2d");
        context.drawImage(image, 0, 0, image.width, image.height);
        const imageData = context.getImageData(0, 0, image.width, image.height).data;
 
        // Check if the image is grayscale
        let isGrayscale = true;
        for (let i = 0; i < imageData.length; i += 4) {
          const r = imageData[i];
          const g = imageData[i + 1];
          const b = imageData[i + 2];
          if (r !== g || g !== b) {
            isGrayscale = false;
            break;
          }
        }
 
        // Update the RGB table with values or display a message for grayscale images
        if (isGrayscale) {
          rgbTableBody.innerHTML = "";
          rgbTableBody.innerHTML = `<tr><td colspan="3">This is a grayscale image.</td></tr>`;
        } else {
          rgbTableBody.innerHTML = "";
          for (let i = 0; i < imageData.length; i += 4) {
            const r = imageData[i];
            const g = imageData[i + 1];
            const b = imageData[i + 2];
 
            const row = document.createElement("tr");
            row.innerHTML = `<td>${r}</td><td>${g}</td><td>${b}</td>`;
            rgbTableBody.appendChild(row);
          }
        }
      }
    });
  });

document.addEventListener("DOMContentLoaded", function () {
   const image = document.querySelector(".img-fluid");
   const imageDimensions = document.getElementById("image-dimensions");
   const rgbTableBody = document.querySelector("#rgb-table tbody");
   const showRgbTableButton = document.getElementById("showRgbTableButton");
   const rgbTable = document.getElementById("rgb-table");

   // Wait for the image to load
   image.addEventListener("load", function () {
     const width = this.width;
     const height = this.height;
     imageDimensions.innerText = `Image Dimensions: ${width} x ${height}`;
   });

   showRgbTableButton.addEventListener("click", function () {
     // Toggle the visibility of the RGB table
     rgbTable.classList.toggle("d-none");

     if (!rgbTable.classList.contains("d-none")) {
       // Get RGB values from the image when the table is visible
       const canvas = document.createElement("canvas");
       const context = canvas.getContext("2d");
       context.drawImage(image, 0, 0, image.width, image.height);
       const imageData = context.getImageData(0, 0, image.width, image.height).data;

       // Check if the image is grayscale
       let isGrayscale = true;
       for (let i = 0; i < imageData.length; i += 4) {
         const r = imageData[i];
         const g = imageData[i + 1];
         const b = imageData[i + 2];
         if (r !== g || g !== b) {
           isGrayscale = false;
           break;
         }
       }

       // Update the RGB table with values or display a message for grayscale images
       if (isGrayscale) {
         rgbTableBody.innerHTML = "";
         rgbTableBody.innerHTML = `<tr><td colspan="3">This is a grayscale image.</td></tr>`;
       } else {
         rgbTableBody.innerHTML = "";
         for (let i = 0; i < imageData.length; i += 4) {
           const r = imageData[i];
           const g = imageData[i + 1];
           const b = imageData[i + 2];

           const row = document.createElement("tr");
           row.innerHTML = `<td>${r}</td><td>${g}</td><td>${b}</td>`;
           rgbTableBody.appendChild(row);
         }
       }
     }
   });
 });
 