"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileText, AlertCircle, Loader2, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useStore } from "@/lib/store";

const ACCEPTED_TYPES = {
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
};

const MAX_SIZE = 50 * 1024 * 1024; // 50MB

export default function SlideUploader() {
  const { upload, isUploading, uploadError, uploadFile } = useStore();

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        uploadFile(acceptedFiles[0]);
      }
    },
    [uploadFile]
  );

  const { getRootProps, getInputProps, isDragActive, fileRejections } =
    useDropzone({
      onDrop,
      accept: ACCEPTED_TYPES,
      maxSize: MAX_SIZE,
      maxFiles: 1,
      disabled: isUploading,
    });

  const rejectionError = fileRejections[0]?.errors[0]?.message;

  // Processing state
  if (upload && upload.status === "PROCESSING") {
    return (
      <div className="flex flex-col items-center gap-4 rounded-2xl border-2 border-brand-200 bg-brand-50 p-12 dark:border-brand-800 dark:bg-brand-950/30">
        <Loader2 className="h-12 w-12 animate-spin text-brand-500" />
        <div className="text-center">
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Processing {upload.filename}...
          </p>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Parsing slides, extracting content, and building your study guide.
          </p>
        </div>
      </div>
    );
  }

  // Ready state
  if (upload && upload.status === "READY") {
    return (
      <div className="flex flex-col items-center gap-4 rounded-2xl border-2 border-green-200 bg-green-50 p-12 dark:border-green-800 dark:bg-green-950/30">
        <CheckCircle2 className="h-12 w-12 text-green-500" />
        <div className="text-center">
          <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            {upload.filename} is ready!
          </p>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            {upload.total_slides} slides parsed and indexed.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div
        {...getRootProps()}
        className={cn(
          "cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-colors",
          isDragActive
            ? "border-brand-400 bg-brand-50 dark:border-brand-500 dark:bg-brand-950/30"
            : "border-gray-300 hover:border-brand-300 hover:bg-gray-50 dark:border-gray-700 dark:hover:border-brand-700 dark:hover:bg-gray-900",
          isUploading && "pointer-events-none opacity-60"
        )}
      >
        <input {...getInputProps()} />

        <div className="flex flex-col items-center gap-4">
          {isUploading ? (
            <Loader2 className="h-12 w-12 animate-spin text-brand-500" />
          ) : (
            <Upload className="h-12 w-12 text-gray-400 dark:text-gray-500" />
          )}

          <div>
            <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {isDragActive
                ? "Drop your file here"
                : isUploading
                  ? "Uploading..."
                  : "Drop your lecture slides here"}
            </p>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              or click to browse. Supports PDF and PPTX (max 50MB).
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs text-gray-400 dark:text-gray-500">
            <span className="flex items-center gap-1">
              <FileText className="h-3.5 w-3.5" /> PDF
            </span>
            <span className="flex items-center gap-1">
              <FileText className="h-3.5 w-3.5" /> PPTX
            </span>
          </div>
        </div>
      </div>

      {(uploadError || rejectionError) && (
        <div className="mt-3 flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
          <AlertCircle className="h-4 w-4 flex-shrink-0" />
          <span>{uploadError || rejectionError}</span>
        </div>
      )}
    </div>
  );
}
