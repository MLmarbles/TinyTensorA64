Observed results:

Before adjacency optimization
Matrix size: 128x128
Time: 2243 average 
Matrix size: 256x256
Time: 19329 average 
Matrix size: 512x512
Time: 227056 average 
Matrix size: 1024x1024
Time: 3398223 average 

After adjacency optimization
Matrix size: 64x64
Time: 420 average 
Matrix size: 128x128
Time: 3905 average 
Matrix size: 256x256
Time: 29206 average 
Matrix size: 512x512
Time: 248029 average 
Matrix size: 1024x1024
Time: 1911844 average 

With tiling 16 chunks:
Matrix size: 64x64
Time: 377 average 
Matrix size: 128x128
Time: 3315 average 
Matrix size: 256x256
Time: 25363 average 
Matrix size: 512x512
Time: 211683 average 
Matrix size: 1024x1024
Time: 1729686 average 

With tiling 32 chunks:
Matrix size: 64x64
Time: 387 average 
Matrix size: 128x128
Time: 3227 average 
Matrix size: 256x256
Time: 25600 average 
Matrix size: 512x512
Time: 206807 average 
Matrix size: 1024x1024
Time: 1711282 average 

With tiling 64 chunks:
Matrix size: 64x64
Time: 362 average 
Matrix size: 128x128
Time: 3171 average 
Matrix size: 256x256
Time: 25946 average 
Matrix size: 512x512
Time: 210594 average 
Matrix size: 1024x1024
Time: 1686869 average 

Using 1 thread worker:
Matrix size: 64x64
Time: 729 average 
Matrix size: 128x128
Time: 5371 average 
Matrix size: 256x256
Time: 38203 average 
Matrix size: 512x512
Time: 314798 average 
Matrix size: 1024x1024
Time: 2628475 average 

Using 2 thread worker:
Matrix size: 64x64
Time: 511 average 
Matrix size: 128x128
Time: 2931 average 
Matrix size: 256x256
Time: 24702 average 
Matrix size: 512x512
Time: 169076 average 
Matrix size: 1024x1024
Time: 1396110 average 

Using 4 thread worker:
Matrix size: 64x64
Time: 589 average 
Matrix size: 128x128
Time: 1992 average 
Matrix size: 256x256
Time: 13823 average 
Matrix size: 512x512
Time: 97548 average 
Matrix size: 1024x1024
Time: 711661 average 

Using 8 thread worker:
Matrix size: 64x64
Time: 644 average 
Matrix size: 128x128
Time: 1756 average 
Matrix size: 256x256
Time: 13846 average 
Matrix size: 512x512
Time: 61568 average 
Matrix size: 1024x1024
Time: 420167 average 
