//! # shrew-data
//!
//! Data loading, datasets, and batching for Shrew.
//!
//! This crate provides:
//! - [`Dataset`] trait — unified interface for any dataset
//! - [`DataLoader`] — batching, shuffling, parallel iteration over a Dataset
//! - [`AsyncDataLoader`] — prefetching data loader with background workers
//   - Dataset combinators — SubsetDataset, ConcatDataset, MapDataset, VecDataset
//   - Image augmentation transforms — RandomFlip, RandomCrop, etc.
//   - CSV dataset loader
//   - Built-in datasets: MNIST (IDX format parser)
//   - ImageFolder (directory-based image classification dataset)
//   - Train/test splitting with reproducible seeding

pub mod async_loader;
pub mod augment;
pub mod combinators;
pub mod csv_dataset;
pub mod dataset;
pub mod image_folder;
pub mod loader;
pub mod mnist;
pub mod transform;

pub use async_loader::{AsyncDataLoader, AsyncDataLoaderConfig, Batch, PrefetchIterator};
pub use augment::{
    ColorJitter, RandomCrop, RandomErasing, RandomHorizontalFlip, RandomNoise, RandomVerticalFlip,
};
pub use combinators::{train_test_split, ConcatDataset, MapDataset, SubsetDataset, VecDataset};
pub use csv_dataset::{CsvConfig, CsvDataset};
pub use dataset::{Dataset, Sample};
pub use loader::{DataLoader, DataLoaderConfig};
pub use mnist::MnistDataset;
pub use transform::Transform;

#[cfg(feature = "image-folder")]
pub use image_folder::{ImageFolder, ImageFolderBuilder};

// Error types for ImageFolder

/// Errors from the ImageFolder dataset.
#[derive(Debug)]
pub enum ImageFolderError {
    /// The root path is not a directory.
    NotADirectory(String),
    /// No class subdirectories found.
    NoClasses(String),
    /// No image files found.
    NoImages(String),
    /// Image decoding failed.
    ImageDecode(String, String),
    /// I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for ImageFolderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageFolderError::NotADirectory(p) => write!(f, "Not a directory: {p}"),
            ImageFolderError::NoClasses(p) => write!(f, "No class subdirectories in {p}"),
            ImageFolderError::NoImages(p) => write!(f, "No image files found in {p}"),
            ImageFolderError::ImageDecode(p, e) => write!(f, "Failed to decode {p}: {e}"),
            ImageFolderError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for ImageFolderError {}
