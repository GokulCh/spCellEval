library(Seurat)
library(class)
library(segmented)
library(readr)
library(tidyverse)
library(future)
library(future.apply)
library(parallelly)
library(TACIT)
library(argparse)

run_TACIT <- function(n, input_path, separate_col, scaling, log1p, decision_matrix_path, r, p, output_path, undo_arcsinh = NULL) {
  data <- read_csv(input_path)
  
  CELLxFEATURE <- data %>% 
    select(cell_id, everything()) %>%
    select(- (separate_col:last_col()))
  feature_cols <- setdiff(names(CELLxFEATURE), "cell_id")
  if (!is.null(scaling)) {
    CELLxFEATURE[feature_cols] <- CELLxFEATURE[feature_cols] * scaling
  }
  if (log1p) {
    CELLxFEATURE[feature_cols] <- log1p(CELLxFEATURE[feature_cols])
  }
  if (!is.null(undo_arcsinh) && undo_arcsinh > 0) {
    # TACIT was designed for (and published on) raw intensities: its microclustering, its per-cell-type score (a weighted SUM of
    # intensities) and its thresholds all assume them. The benchmark table holds arcsinh(x / cofactor), which is exactly invertible.
    CELLxFEATURE[feature_cols] <- undo_arcsinh * sinh(CELLxFEATURE[feature_cols])
    message(sprintf("TACIT input: arcsinh undone with cofactor %g (raw intensities %g..%g)", undo_arcsinh,
                    min(unlist(CELLxFEATURE[feature_cols])), max(unlist(CELLxFEATURE[feature_cols]))))
  }
  data <- data %>%
    rename(true_phenotype = cell_type)
  
  TYPExMARKER <- read_csv(decision_matrix_path)
  inference_times <- numeric(n)
  message(sprintf("TACIT: %d cells, %d markers, %d cell types, r=%s p=%s", nrow(CELLxFEATURE), length(feature_cols), nrow(TYPExMARKER), r, p))
  for(i in 1:n) {
    start_time <- Sys.time()
    TACIT_result <- TACIT(CELLxFEATURE, TYPExMARKER, r=r, p=p)
    end_time <- Sys.time()
    elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
    inference_times[i] <- elapsed
    
    data <- data %>%
      mutate(predicted_phenotype = TACIT_result$TACIT) %>%
      mutate(predicted_phenotype = case_when(
        predicted_phenotype == "Others" ~ "undefined",
        TRUE ~ predicted_phenotype
      ))
    
    output_file <- file.path(output_path, paste0("predictions_", i, ".csv"))
    write_csv(data, output_file)
  }
  timing_file <- file.path(output_path, "fold_times.txt")
  file_conn <- file(timing_file, "w")
  for (i in 1:n) {
    writeLines(paste0("Fold ", i, " inference_time: ", round(inference_times[i], 2), " seconds"), file_conn)
  }
  close(file_conn)
}

main <- function() {
  parser <- ArgumentParser(description = "Run TACIT phenotype predictions")
  
  parser$add_argument("--input_path", type="character", required=TRUE,
                      help="Path to input quantification CSV file")
  parser$add_argument("--decision_matrix_path", type="character", required=TRUE,
                      help="Path to marker decision matrix CSV file")
  parser$add_argument("--separate_col", type="character", required=TRUE,
                      help="Col that separates marker columns from remaining cols. Input like this from terminal: 'colname' ")
  parser$add_argument("--scaling", type="double", default=NULL,
                      help="Scaling factor for the data. If not provided, no scaling is applied")
  parser$add_argument("--log1p", action='store_true', help="Apply log1p transformation to the data")
  parser$add_argument("-n", "--iterations", type="integer", default=5,
                      help="Number of times to run the TACIT prediction [default %(default)s]")
  parser$add_argument("-r", type="integer", default=10,
                      help="resolution. The higher the resolution, the greaterde the number of microclusters")
  parser$add_argument("-p", type="integer", default=10,
                      help="Dimension. TNumber of dimensions used for microclusters")
  parser$add_argument("--output_path", type="character", required=TRUE,
                      help="Directory to save prediction CSV files")
  parser$add_argument("--undo_arcsinh", type="double", default=NULL,
                      help="the table holds arcsinh(x / COFACTOR): convert back to raw intensities first (TACIT's published setting)")
  parser$add_argument("--use_repo_source", action="store_true",
                      help="use the TACIT code vendored in <repo>/TACIT/R instead of the installed package (they can differ in version)")
  args <- parser$parse_args()
  message("installed TACIT package: ", tryCatch(as.character(packageVersion("TACIT")), error = function(e) "unknown"),
          "; TACIT() arguments: ", paste(names(formals(TACIT)), collapse = ", "))
  if (isTRUE(args$use_repo_source)) {
    f <- file.path(dirname(normalizePath(sub("--file=", "", commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]))),
                   "..", "..", "..", "TACIT", "R", "TACIT_function.R")
    tryCatch({ source(normalizePath(f)); message("using the vendored TACIT code: ", normalizePath(f)) },
             error = function(e) message("could not source ", f, " (", conditionMessage(e), "); using the installed package"))
  }
  options(future.globals.maxSize = 6 * 1024^3)
  run_TACIT(args$iterations, args$input_path,args$separate_col, args$scaling, args$log1p, args$decision_matrix_path, args$r, args$p, args$output_path, args$undo_arcsinh)
}

main()