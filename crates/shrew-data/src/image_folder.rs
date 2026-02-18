// ImageFolder — Directory-based image classification dataset
//
// Loads images from a directory structure where each subdirectory is a class:
//
//   root/
//     class_a/
//       img_001.png
//       img_002.jpg
//     class_b/
//       img_003.png
//       ...
//
// Class labels are assigned as sorted indices of subdirectory names.
//
// The dataset returns samples with:
//   - features: pixel values in [C, H, W] layout, normalised to [0, 1]
//   - feature_shape: [C, H, W]
//   - target: [class_index as f64]
//   - target_shape: [1]
//
// USAGE:
//
//   let ds = ImageFolder::new("data/imagenet/train")
//       .resize(224, 224)
//       .build()?;
//   println!("{} images, {} classes", ds.len(), ds.class_names().len());
//
// Requires the `image-folder` feature (which brings in the `image` crate).

#[cfg(feature = "image-folder")]
pub use inner::*;

#[cfg(feature = "image-folder")]
mod inner {
    use std::path::{Path, PathBuf};

    use image::imageops::FilterType;
    use image::GenericImageView;

    use crate::dataset::{Dataset, Sample};

    /// Supported image extensions (case-insensitive).
    const EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "gif", "tiff", "tif", "webp"];

    fn is_image(path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
            .unwrap_or(false)
    }

    // ImageFolderConfig (builder)

    /// Builder for [`ImageFolder`].
    pub struct ImageFolderBuilder {
        root: PathBuf,
        resize: Option<(u32, u32)>,
        grayscale: bool,
    }

    impl ImageFolderBuilder {
        /// Create a builder rooted at the given directory.
        pub fn new<P: AsRef<Path>>(root: P) -> Self {
            ImageFolderBuilder {
                root: root.as_ref().to_path_buf(),
                resize: None,
                grayscale: false,
            }
        }

        /// Resize all images to (width, height) using Lanczos3 filter.
        pub fn resize(mut self, width: u32, height: u32) -> Self {
            self.resize = Some((width, height));
            self
        }

        /// Convert images to grayscale (1 channel instead of 3).
        pub fn grayscale(mut self, yes: bool) -> Self {
            self.grayscale = yes;
            self
        }

        /// Scan the directory tree and build the dataset.
        pub fn build(self) -> Result<ImageFolder, crate::ImageFolderError> {
            ImageFolder::scan(self.root, self.resize, self.grayscale)
        }
    }

    // ImageFolder dataset

    /// A directory-based image classification dataset (like torchvision ImageFolder).
    #[derive(Debug)]
    pub struct ImageFolder {
        /// Sorted class names (subdirectory names).
        class_names: Vec<String>,
        /// Per-sample metadata: (path, class_index).
        entries: Vec<(PathBuf, usize)>,
        /// Optional resize target (width, height).
        resize: Option<(u32, u32)>,
        /// Whether to convert to grayscale.
        grayscale: bool,
        /// Number of channels (1 if grayscale, 3 otherwise).
        channels: usize,
        /// Image width after optional resize (0 if no resize — varies per image).
        width: u32,
        /// Image height after optional resize.
        height: u32,
    }

    impl ImageFolder {
        /// Convenience entry-point: `ImageFolder::new(root)` returns a builder.
        pub fn new<P: AsRef<Path>>(root: P) -> ImageFolderBuilder {
            ImageFolderBuilder::new(root)
        }

        /// Scan the directory and collect all image paths + class labels.
        fn scan(
            root: PathBuf,
            resize: Option<(u32, u32)>,
            grayscale: bool,
        ) -> Result<Self, crate::ImageFolderError> {
            if !root.is_dir() {
                return Err(crate::ImageFolderError::NotADirectory(
                    root.display().to_string(),
                ));
            }

            // Collect class subdirectories (sorted)
            let mut class_dirs: Vec<(String, PathBuf)> = Vec::new();
            for entry in std::fs::read_dir(&root).map_err(|e| crate::ImageFolderError::Io(e))? {
                let entry = entry.map_err(|e| crate::ImageFolderError::Io(e))?;
                let path = entry.path();
                if path.is_dir() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        class_dirs.push((name.to_string(), path));
                    }
                }
            }
            class_dirs.sort_by(|a, b| a.0.cmp(&b.0));

            if class_dirs.is_empty() {
                return Err(crate::ImageFolderError::NoClasses(
                    root.display().to_string(),
                ));
            }

            let class_names: Vec<String> = class_dirs.iter().map(|(n, _)| n.clone()).collect();

            // Collect image paths per class
            let mut entries: Vec<(PathBuf, usize)> = Vec::new();
            for (class_idx, (_name, dir)) in class_dirs.iter().enumerate() {
                let mut paths: Vec<PathBuf> = Vec::new();
                Self::collect_images(dir, &mut paths);
                paths.sort();
                for p in paths {
                    entries.push((p, class_idx));
                }
            }

            if entries.is_empty() {
                return Err(crate::ImageFolderError::NoImages(
                    root.display().to_string(),
                ));
            }

            let channels = if grayscale { 1 } else { 3 };
            let (width, height) = resize.unwrap_or((0, 0));

            Ok(ImageFolder {
                class_names,
                entries,
                resize,
                grayscale,
                channels,
                width,
                height,
            })
        }

        /// Recursively collect image files.
        fn collect_images(dir: &Path, out: &mut Vec<PathBuf>) {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for entry in rd.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        Self::collect_images(&path, out);
                    } else if is_image(&path) {
                        out.push(path);
                    }
                }
            }
        }

        /// Get the class names (sorted).
        pub fn class_names(&self) -> &[String] {
            &self.class_names
        }

        /// Number of classes.
        pub fn num_classes(&self) -> usize {
            self.class_names.len()
        }

        /// Get the class index for the i-th sample.
        pub fn class_of(&self, index: usize) -> usize {
            self.entries[index].1
        }

        /// Get the file path of the i-th sample.
        pub fn path_of(&self, index: usize) -> &Path {
            &self.entries[index].0
        }

        /// Load and decode an image, returning pixel data in [C, H, W] layout
        /// with values normalised to [0, 1].
        fn load_image(
            &self,
            index: usize,
        ) -> Result<(Vec<f64>, [usize; 3]), crate::ImageFolderError> {
            let path = &self.entries[index].0;
            let img = image::open(path).map_err(|e| {
                crate::ImageFolderError::ImageDecode(path.display().to_string(), e.to_string())
            })?;

            // Optional resize
            let img = match self.resize {
                Some((w, h)) => img.resize_exact(w, h, FilterType::Lanczos3),
                None => img,
            };

            // Grayscale or RGB
            let (w, h) = img.dimensions();
            let (pixels, c) = if self.grayscale {
                let gray = img.to_luma8();
                let data: Vec<f64> = gray.as_raw().iter().map(|&v| v as f64 / 255.0).collect();
                (data, 1usize)
            } else {
                let rgb = img.to_rgb8();
                let raw = rgb.as_raw();
                // Convert from [H, W, C] interleaved to [C, H, W] planar
                let npix = (w * h) as usize;
                let mut data = vec![0.0f64; 3 * npix];
                for i in 0..npix {
                    data[i] = raw[i * 3] as f64 / 255.0; // R
                    data[npix + i] = raw[i * 3 + 1] as f64 / 255.0; // G
                    data[2 * npix + i] = raw[i * 3 + 2] as f64 / 255.0; // B
                }
                (data, 3usize)
            };

            Ok((pixels, [c, h as usize, w as usize]))
        }
    }

    impl Dataset for ImageFolder {
        fn len(&self) -> usize {
            self.entries.len()
        }

        fn get(&self, index: usize) -> Sample {
            match self.load_image(index) {
                Ok((features, shape)) => Sample {
                    features,
                    feature_shape: shape.to_vec(),
                    target: vec![self.entries[index].1 as f64],
                    target_shape: vec![1],
                },
                Err(e) => {
                    // Return a zero sample on error (avoids panicking in Iterator)
                    let c = self.channels;
                    let (w, h) = self.resize.unwrap_or((1, 1));
                    eprintln!(
                        "ImageFolder: failed to load {:?}: {}",
                        self.entries[index].0, e
                    );
                    Sample {
                        features: vec![0.0; c * (h as usize) * (w as usize)],
                        feature_shape: vec![c, h as usize, w as usize],
                        target: vec![self.entries[index].1 as f64],
                        target_shape: vec![1],
                    }
                }
            }
        }

        fn feature_shape(&self) -> &[usize] {
            // Only valid when resize is set; otherwise shape varies per image.
            // We return a static reference to a leaked slice for the fixed case.
            // For dynamic case, we return a placeholder.
            &[]
        }

        fn target_shape(&self) -> &[usize] {
            &[]
        }

        fn name(&self) -> &str {
            "ImageFolder"
        }
    }

    // Send + Sync — all fields are owned data
    unsafe impl Send for ImageFolder {}
    unsafe impl Sync for ImageFolder {}
}
