/* L-10 MCS 572 Wed 1 Feb 2023 : comptrap_omp.c
 * The composite trapezoidal rule on sqrt(1-x^2) over [0,1] to compute pi,
 * with OpenMP illustrating the parallel construct with a private clause
 * and a critical section for the update of the approximation for pi. */

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#define v 0      /* verbose flag */

double traprule ( double (*f) ( double x ), double a, double b, int n );
/* applies the composite Trapezoidal rule to approximate the integral
 * of f over [a,b] using n+1 function evaluations, use only for n > 0 */

double integrand ( double x ); /* the function we integrate */

int main ( int argc, char *argv[] )
{
   int i, p = 0, n = 1000000;
   double my_pi = 0.0;
   double a,b,c,h,y,pi,error;

   if(argc == 1)
   {
      printf("Please specify the number threads,\n");
      printf("as the first argument of the program.\n");
      return 1;
   }
   else
   {
      p = atoi(argv[1]);
      omp_set_num_threads(p);
   }

   h = 1.0/p;

   #pragma omp parallel private(i,a,b,c)
   /* each thread has its own i, a, b, and c */
   {
      i = omp_get_thread_num();
      a = i*h;
      b = (i+1)*h;

      if(v>0)
      {
         printf("Thread %d integrates from %.2e to %.2e\n",i,a,b);
         fflush(stdout);
      }
      c = traprule(integrand,a,b,n);
      #pragma omp critical    /* critical section protects shared my_pi */
         my_pi += c;

      if(v>0)
      {
         printf("Thread %d computes %.15e as approximation.\n",i,my_pi);
         fflush(stdout);
      }
   }
   my_pi = 4.0*my_pi; pi = 2.0*asin(1.0); error = my_pi-pi;

   printf("Approximation for pi = %.15e with error = %.3e\n",my_pi,error);

   return 0;
}

double integrand ( double x )
{
   return sqrt(1.0 - x*x);
}

double traprule ( double (*f) ( double x ), double a, double b, int n )
{
   int i;
   double h = (b-a)/n; 
   double y = (f(a) + f(b))/2.0;
   double x;

   for(i=1,x=a+h; i < n; i++,x+=h) y += f(x);

   return h*y;
}
