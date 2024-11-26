#include <cblas.h>
#include <immintrin.h>
#include <eigen3/Eigen/Dense>

extern "C" {
  /* void gemm_c(double* A, double* B, double* C, size_t rows, size_t cols, size_t inner){ */
  /*   for(size_t i = 0; i<rows; i++){ */
  /*     for(size_t j = 0; j<cols; j++){ */
  /*       double acc = 0.0; */
  /*       for(size_t k = 0; k<inner; k++){ */
  /*         acc += A[cols * i + k] * B[rows * k + j]; */
  /*       } */
  /*       C[cols * i + j] = acc; */
  /*     } */
  /*   } */
  /* } */


  void gemm_c(double* A, double* B, double* C, size_t rows, size_t cols, size_t inner) {
      const size_t block_size = 32;  // Adjust the block size based on your architecture

      for (size_t i = 0; i < rows; i += block_size) {
          for (size_t j = 0; j < cols; j += block_size) {
              for (size_t ii = i; ii < i + block_size && ii < rows; ii++) {
                  for (size_t jj = j; jj < j + block_size && jj < cols; jj++) {
                      __m256d acc = _mm256_setzero_pd();

                      for (size_t k = 0; k < inner; k += 4) {
                          __m256d a_vec = _mm256_loadu_pd(&A[ii * inner + k]);
                          __m256d b_vec = _mm256_loadu_pd(&B[k * cols + jj]);
                          acc = _mm256_add_pd(acc, _mm256_mul_pd(a_vec, b_vec));
                      }

                      // Horizontal sum of the vector
                      __m128d acc_low = _mm256_castpd256_pd128(acc);
                      __m128d acc_high = _mm256_extractf128_pd(acc, 1);
                      __m128d sum = _mm_add_pd(acc_low, acc_high);

                      // Store the result
                      C[ii * cols + jj] = _mm_cvtsd_f64(sum);
                  }
              }
          }
      }
  }

  void gemm_blas(double* A, double* B, double* C, size_t rows, size_t cols, size_t inner){
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        rows, cols, inner, 1.0, A, inner, B, cols, 0.0, C, cols
    );
  }

  void gemm_eigen(double* A, double* B, double* C, size_t rows, size_t cols, size_t inner){
    Eigen::Map<const Eigen::MatrixXd> A_eigen(A, rows, inner);
    Eigen::Map<const Eigen::MatrixXd> B_eigen(B, inner, cols);
    Eigen::Map<Eigen::MatrixXd> C_eigen(C, rows, cols);

    C_eigen = A_eigen * B_eigen;
  }
}
