#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/**
 * C BLAS function wrappers.
 */

inline void copy_vector_cblas(
  const double* const x, double* const y,
  const unsigned int rows )
{
  cblas_dcopy( rows, x, 1, y, 1 );
}

inline double inner_product_cblas(
  const double* const x, const double* const y,
  const unsigned int rows )
{
  return cblas_ddot( rows, x, 1, y, 1 );
}

inline void outer_product_cblas(
  const double* const x, const double* const y,
  const double alpha, const unsigned int rows_x, const unsigned int rows_y,
  double* m )
{
  cblas_dger( CblasRowMajor, rows_x, rows_y, alpha, x, 1, y, 1, m, rows_y);
}

inline void vector_vector_sum_cblas(
  const double* const x, double* const y,
  const double alpha, const unsigned int rows )
{
  cblas_daxpy( rows, alpha, x, 1, y, 1);
}

/**
 * Helper functions.
 */

inline void copy_matrix(
  const double* const m0, double* const m1,
  const unsigned int rows, const unsigned int cols )
{
  for ( unsigned int i = 0; i < rows; ++i )
  {
    for ( unsigned int j = 0; j < cols; ++j )
    {
      m1[ i * cols + j ] = m0[ i * cols + j ];
    }
  }
}

inline void matrix_matrix_sum(
  const double* const m0, const double* const m1,
  const double alpha, const unsigned int rows, const unsigned int cols,
  double* m2 )
{
  for ( unsigned int i = 0; i < rows; ++i )
  {
    for ( unsigned int j = 0; j < cols; ++j )
    {
      m2[ i * cols + j ] = m0[ i * cols + j ] + alpha * m1[ i * cols + j ];
    }
  }
}

inline void zero_vector(
  const unsigned int rows,
  double* x )
{
  memset(x, 0, rows * sizeof( double ) );
}

inline void zero_matrix(
  const unsigned int rows, const unsigned int cols,
  double* m )
{
  memset(m, 0, rows * cols * sizeof( double ) );
}

inline void zero_diagonal(
  const unsigned int rows, const unsigned int cols,
  double* m )
{
  for ( unsigned int i = 0; i < rows; ++i )
  {
    m[ i * cols + i ] = 0;
  }
}

inline double rand_double( void)
{
  return ( double ) rand() / ( double )( RAND_MAX );
}

inline unsigned int logistic( const double h )
{
  return 1. / ( 1. + exp( -h ) ) > rand_double();
}

/**
 * Train a fully visible Boltzmann machine with CD-1.
 */
void ctrain(
  double* W, double* b, unsigned int rows, unsigned int cols,
  double* data, unsigned int episodes,
  double epsilon_w, double epsilon_b,
  unsigned int batchsize, unsigned int seed,
  const double momentum_constant, const double decay_constant )
{
  double dW[ rows * cols ];
  double db[ cols ];
  double dW_minus[ rows * cols ];
  double db_minus[ cols ];
  double u;

  const double effective_epsilon_w = 1 / ( double )( batchsize ) * epsilon_w;
  const double effective_epsilon_b = 1 / ( double )( batchsize ) * epsilon_b;

  srand( seed );

  // initialize momentum terms to zero at start of learning
  zero_matrix( rows, cols, dW_minus );
  zero_vector( rows, db_minus );

  for ( unsigned int i = 0; i < episodes / batchsize; ++i )
  {
    // initialize weights and bias change to zero at start of batch
    zero_matrix( rows, cols, dW );
    zero_vector( rows, db );

    for ( unsigned int j = 0; j < batchsize; ++j )
    {
      // compute \Delta W += <ss>_0
      outer_product_cblas( &data[ i * batchsize * cols + j * cols], &data[ i * batchsize * cols + j * cols], 1., rows, cols, dW );

      // compute \Delta b += <s>_0
      vector_vector_sum_cblas( &data[ i * batchsize * cols + j * cols], db, 1, rows );

      // perform a single Gibbs step (update all units once) directly
      // on data vector to avoid copying
      for ( unsigned int k = 0; k < cols; ++k )
      {
        u = inner_product_cblas( &W[ k * cols ], &data[ i * batchsize * cols + j * cols], cols );
        data[ i * batchsize * cols + j * cols + k ] = logistic( u + b[ k ] );
      }

      // compute \Delta W -= <ss>_1
      outer_product_cblas( &data[ i * batchsize * cols + j * cols ], &data[ i * batchsize * cols + j * cols ], -1., rows, cols, dW );

      // compute \Delta b -= <s>_1
      vector_vector_sum_cblas( &data[ i * batchsize * cols + j * cols ], db, -1, rows );
    }

    // add momentum
    if ( momentum_constant > 0. )
    {
      matrix_matrix_sum( dW, dW_minus, momentum_constant, rows, cols, dW );
      vector_vector_sum_cblas( db_minus, db, momentum_constant, rows );
      copy_matrix( dW, dW_minus, rows, cols );
      copy_vector_cblas( db, db_minus, rows );
    }

    // decay weights
    if ( decay_constant > 0. )
    {
      // need to devide by learning rate, since dW is multiplied by it
      // below
      matrix_matrix_sum( dW, W, -decay_constant / effective_epsilon_w, rows, cols, dW );
    }

    // updates weights and biases
    matrix_matrix_sum( W, dW, effective_epsilon_w, rows, cols, W );
    vector_vector_sum_cblas( db, b, effective_epsilon_b, rows );

    // set diagonal of weight matrix to zero
    zero_diagonal( rows, cols, W );
  }
}

/**
 * Sample from a fully visible Boltzmann machine.
 */
void csample(
  const double* W, const double* b, const unsigned int rows, const unsigned int cols,
  const unsigned int episodes, const unsigned int seed,
  double* s, double* samples )
{
  double u;
  srand( seed );

  unsigned int i = 0;
  while( i < episodes )
  {
    // perform a single Gibbs step (update all units once)
    for ( unsigned int k = 0; k < cols; ++k )
    {
      u = inner_product_cblas( &W[ k * cols ], s, cols );
      s[ k ] = logistic( u + b[ k ] );
      copy_vector_cblas( s, &samples[ i * rows], rows );
      ++i;
    }
  }
}
