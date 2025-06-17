use flate2::read::GzDecoder;
use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use tar::Archive;

const PDFIUM_BASE_URL: &str =
    "https://github.com/bblanchon/pdfium-binaries/releases/download/chromium/6721";

struct PdfiumBinary {
    filename: String,
    platform: String,
    arch: String,
}

fn get_pdfium_binary() -> Result<PdfiumBinary, String> {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| String::from("unknown"));
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| String::from("unknown"));

    match (target_os.as_str(), target_arch.as_str()) {
        ("macos", "aarch64") => Ok(PdfiumBinary {
            filename: "pdfium-mac-arm64.tgz".to_string(),
            platform: "mac".to_string(),
            arch: "arm64".to_string(),
        }),
        ("macos", "x86_64") => Ok(PdfiumBinary {
            filename: "pdfium-mac-x64.tgz".to_string(),
            platform: "mac".to_string(),
            arch: "x64".to_string(),
        }),
        ("linux", "aarch64") => Ok(PdfiumBinary {
            filename: "pdfium-linux-arm64.tgz".to_string(),
            platform: "linux".to_string(),
            arch: "arm64".to_string(),
        }),
        ("linux", "arm") => Ok(PdfiumBinary {
            filename: "pdfium-linux-arm.tgz".to_string(),
            platform: "linux".to_string(),
            arch: "arm".to_string(),
        }),
        ("linux", "x86_64") => Ok(PdfiumBinary {
            filename: "pdfium-linux-x64.tgz".to_string(),
            platform: "linux".to_string(),
            arch: "x64".to_string(),
        }),
        ("linux", "x86") => Ok(PdfiumBinary {
            filename: "pdfium-linux-x86.tgz".to_string(),
            platform: "linux".to_string(),
            arch: "x86".to_string(),
        }),
        ("android", "aarch64") => Ok(PdfiumBinary {
            filename: "pdfium-android-arm64.tgz".to_string(),
            platform: "android".to_string(),
            arch: "arm64".to_string(),
        }),
        ("android", "arm") => Ok(PdfiumBinary {
            filename: "pdfium-android-arm.tgz".to_string(),
            platform: "android".to_string(),
            arch: "arm".to_string(),
        }),
        ("android", "x86_64") => Ok(PdfiumBinary {
            filename: "pdfium-android-x64.tgz".to_string(),
            platform: "android".to_string(),
            arch: "x64".to_string(),
        }),
        ("android", "x86") => Ok(PdfiumBinary {
            filename: "pdfium-android-x86.tgz".to_string(),
            platform: "android".to_string(),
            arch: "x86".to_string(),
        }),
        ("ios", "aarch64") => {
            let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_else(|_| String::from(""));
            if target_env == "simulator" {
                Ok(PdfiumBinary {
                    filename: "pdfium-ios-simulator-arm64.tgz".to_string(),
                    platform: "ios".to_string(),
                    arch: "simulator-arm64".to_string(),
                })
            } else {
                Ok(PdfiumBinary {
                    filename: "pdfium-ios-device-arm64.tgz".to_string(),
                    platform: "ios".to_string(),
                    arch: "device-arm64".to_string(),
                })
            }
        }
        ("ios", "x86_64") => Ok(PdfiumBinary {
            filename: "pdfium-ios-simulator-x64.tgz".to_string(),
            platform: "ios".to_string(),
            arch: "simulator-x64".to_string(),
        }),
        _ => Err(format!(
            "Unsupported platform: {} on {}",
            target_os, target_arch
        )),
    }
}

fn create_build_directories(out_dir: &Path) -> Result<PathBuf, String> {
    let pdfium_dir = out_dir.join("pdfium");
    fs::create_dir_all(&pdfium_dir)
        .map_err(|e| format!("Failed to create pdfium directory: {}", e))?;
    Ok(pdfium_dir)
}

fn download_pdfium_binary(binary_info: &PdfiumBinary, tgz_path: &PathBuf) -> Result<(), String> {
    let url = format!("{}/{}", PDFIUM_BASE_URL, binary_info.filename);
    println!(
        "Downloading PDFium binary for {}-{} from {}...",
        binary_info.platform, binary_info.arch, url
    );

    let response = reqwest::blocking::get(&url)
        .map_err(|e| format!("Failed to download pdfium library: {}", e))?;

    if !response.status().is_success() {
        return Err(format!(
            "Failed to download pdfium library from {}: HTTP status {}",
            url,
            response.status()
        ));
    }

    let bytes = response
        .bytes()
        .map_err(|e| format!("Failed to get response bytes: {}", e))?;
    fs::write(tgz_path, &bytes)
        .map_err(|e| format!("Failed to write pdfium archive to file: {}", e))?;

    println!("Downloaded PDFium binary successfully.");
    Ok(())
}

fn extract_pdfium_archive(tgz_path: &PathBuf, pdfium_dir: &PathBuf) -> Result<(), String> {
    println!("Extracting PDFium archive...");
    let tar_gz = File::open(tgz_path).map_err(|e| format!("Failed to open tarball file: {}", e))?;
    let tar = GzDecoder::new(tar_gz);
    let mut archive = Archive::new(tar);

    archive
        .unpack(pdfium_dir)
        .map_err(|e| format!("Failed to extract pdfium library: {}", e))?;

    println!("Extraction completed.");
    Ok(())
}

fn get_workspace_dir() -> Result<PathBuf, String> {
    let workspace_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR")
            .map_err(|e| format!("Failed to get CARGO_MANIFEST_DIR: {}", e))?,
    );
    workspace_dir
        .parent()
        .ok_or_else(|| String::from("Failed to find parent directory of CARGO_MANIFEST_DIR"))
        .map(|p| p.to_path_buf())
}

fn should_download_pdfium() -> Result<bool, Box<dyn Error>> {
    let workspace_lib_dir = get_workspace_dir()?.join("lib");
    Ok(!workspace_lib_dir.exists() || fs::read_dir(&workspace_lib_dir)?.next().is_none())
}

fn move_to_workspace_lib(pdfium_dir: &Path) -> Result<(), String> {
    println!("Moving PDFium library to workspace lib directory...");
    let lib_path = pdfium_dir.join("lib");
    let workspace_lib_dir = get_workspace_dir()?.join("lib");

    fs::remove_dir_all(&workspace_lib_dir).ok();
    fs::rename(lib_path, workspace_lib_dir)
        .map_err(|e| format!("Failed to move library to workspace: {}", e))?;

    println!("Library moved successfully.");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting PDFium library download and setup...");
    let binary_info = get_pdfium_binary()?;
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let pdfium_dir = create_build_directories(&out_dir)?;
    let tgz_path = out_dir.join("pdfium.tgz");

    if should_download_pdfium()? {
        download_pdfium_binary(&binary_info, &tgz_path)?;
        extract_pdfium_archive(&tgz_path, &pdfium_dir)?;
        move_to_workspace_lib(&pdfium_dir)?;
    } else {
        println!("PDFium library already exists in workspace/lib. Skipping download.");
    }

    println!("PDFium setup completed successfully!");
    Ok(())
}
