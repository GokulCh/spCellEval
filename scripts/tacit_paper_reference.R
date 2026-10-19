#!/usr/bin/env Rscript
# Reference run: the TACIT developers' own CRC pipeline (TACIT/Code generate figure/runTACIT_paper.R) on the RAW expression file, scored against
# the original ClusterName labels the same way the benchmark scores TACIT (macro-F1 over true + predicted labels; "Others" counts as wrong).
# Use it to tell "TACIT is limited on this data" from "our benchmark integration of TACIT is the problem".
#
# Run from the repo root:
#   Rscript scripts/tacit_paper_reference.R [raw_csv] [signature_csv] [out_dir] [r] [p]
# defaults: data/datasets/crc_tma/raw/CRC_TMA_expression.csv  src/methods/TACIT/crc_tma_decision_matrix_level3.csv  data/results/tacit_reference  200  20
# Writes <out_dir>/tacit_reference_predictions.csv, tacit_reference_per_class.csv and prints accuracy + macro-F1.
suppressPackageStartupMessages({ library(readr); library(dplyr) })

a <- commandArgs(trailingOnly = TRUE)
arg <- function(i, d) if (length(a) >= i && nzchar(a[i])) a[i] else d
raw_csv <- arg(1, "data/datasets/crc_tma/quantification/raw/CRC_TMA_expression.csv")
sig_csv <- arg(2, "src/methods/TACIT/crc_tma_decision_matrix_level3.csv")
out_dir <- arg(3, "data/results/tacit_reference")
r <- as.integer(arg(4, "200")); p <- as.integer(arg(5, "20"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

source("TACIT/R/TACIT_function.R")          # the vendored (current) TACIT code; defines TACIT(data_expression, r, p, Signature)

data <- read_csv(raw_csv, show_col_types = FALSE)
Signature <- read_csv(sig_csv, show_col_types = FALSE)

# the paper's quality control: merge sub-names, keep only the cell types that have a signature
macro <- c("CD11b+ CD68+ macrophages", "CD163+ macrophages", "CD68+ macrophages", "CD68+ macrophages GzmB+", "CD68+ CD163+ macrophages")
data$ClusterName <- ifelse(data$ClusterName %in% macro, "macrophages", data$ClusterName)
data$ClusterName <- ifelse(data$ClusterName %in% "CD4+ T cells GATA3+", "CD4+ T cells", data$ClusterName)
data <- data[data$ClusterName %in% Signature$cell_type, ]
Signature[is.na(Signature)] <- 0
Signature <- as.data.frame(Signature)
missing <- setdiff(colnames(Signature)[-1], colnames(data))
if (length(missing)) stop("signature columns not found in the raw file: ", paste(missing, collapse = ", "))
message(sprintf("%d cells, %d signature markers, %d cell types; TACIT r=%d p=%d on RAW intensities", nrow(data), ncol(Signature) - 1, nrow(Signature), r, p))

t0 <- Sys.time()
res <- TACIT(data_expression = as.data.frame(data), r = r, p = p, Signature = Signature)
message(sprintf("TACIT finished in %.1f min", as.numeric(difftime(Sys.time(), t0, units = "mins"))))

truth <- as.character(data$ClusterName)
pred <- as.character(res$TACIT)
write_csv(data.frame(CellID = data$CellID, truth = truth, predicted = pred), file.path(out_dir, "tacit_reference_predictions.csv"))

labs <- union(truth, pred)
per <- do.call(rbind, lapply(labs, function(l) {
  tp <- sum(truth == l & pred == l); fp <- sum(truth != l & pred == l); fn <- sum(truth == l & pred != l)
  data.frame(cell_type = l, support = sum(truth == l), n_predicted = sum(pred == l), f1 = if (2 * tp + fp + fn) 2 * tp / (2 * tp + fp + fn) else 0)
}))
per <- per[order(-per$support), ]
write_csv(per, file.path(out_dir, "tacit_reference_per_class.csv"))
cat(sprintf("\naccuracy        %.3f\nmacro-F1        %.3f   (over %d labels: true + predicted)\nunassigned      %.3f   (\"Others\" / empty)\n",
            mean(truth == pred), mean(per$f1), length(labs), mean(pred %in% c("Others", ""))))
print(per, row.names = FALSE)
