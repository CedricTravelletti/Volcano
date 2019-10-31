# Niklas's Data
Received on 22 November 2018.


## land-based gravity data from Stromboli
**d_land** in mGal containing 543 observations (marine data removed).

### Corrections applied:

Corrected for bathymetric effects, tide, free-air (distance from the Earth's center),
latitude, instrument drift, and a large-scale trend related to
deeper lithospheric structure that we are not interested in.

The data are relative variations with respect to a base station (datum 1).
For the errors, you can assume a Gaussian error with a standard deviation of 0.1 mGal.
In my work, I used iterative-reweighed-least-squares, which implied that I rather
assumed a symmetric exponential. 
However, I don't think the distinction is crucial at this point.

The forward operator **F_land** will allow you to simulate data,

$$d_{sim} = F_{land} * m$$, where m is a vector of densities (kg/m^3)
for each of the model cells.

This operator accounts for the fact that the measurements are relative to
a base station, so **d_data** and **dsim** should match within 1 mGal
if you have a good model m.
The matrix $F_{land}$ has 543 columns and 179171 rows (i.e., the number of model parameters considered).

### Topography and Location
There are two digital elevation models: x,y,z are obtained from a 10 by 10 m
resolution model, while $x_{fine}$, $y_{fine}$ and $z_{fine}$ is from
a highly accurate 1 by 1 m
resolution model (however, its extent is smaller).

You can use this DEM model,
but we can't distribute it. The DEM can be used for visualisation purposes or
you might want to use it if you develop your own forward solver.

$long$ is the longitude of the measurement locations, $lat$ is their latitude,
$h_{true}$ is the height of the instrument, $GPS_{ground}$ is the measured height
at the surface.

$x_i$, $y_i$, $z_i$ are the midpoints when I transform my inversion results to a regular
grid for visualisation purposes

$dx_i$, $dy_i$ and $sep_z$ are the corresponding discretisation sizes

However, many of these cells are located in the air
(so they do not exist).
The first column in **exist** indicates if a given cell in the regular grid
defined by $x_i$, $y_i$, and $z_i$ exists by a 1 (0 if it is in the air).
The second column provides the link to matrix $F_{land}$ by giving
the corresponding column in the $F_{land}$ matrix.
The order in exist is such that x cycles the fastest,
followed by y, followed by z.

Finally, $ind$ gives for each column in $F_{land}$, the z, y and x index
in relation to the $z_i$, $y_i$ and $x_i$ values.

### Variables
We have $n_{obs}$ = 543 observations and $n_{model} = 179171$ model parameter (cells).

* **F_land** Group holding several vars.
**F_land/data** contains the actual forward operator. It should have $n_{obs}$
columns and $n_{model}$ rows, but it was flattened as a list (line by line).


* **d_land** Observations (corrected) shape (1, $n_{obs}$).

* **GPS_ground** shape (1, $n_{obs}$).

* **h_true** shape (1, $n_{obs}$).

* **lat** shape (1, $n_{obs}$).

* **long** shape (1, $n_{obs}$).


#### Inversion Grid
The results are computed on a different grid, defined by xi, yi, zi.

* **x_i** (196, 1) longitude of the result grid (midpoint).
* **dxi** (196, 1) size of the result grid.

* **yi** (192, 1) latitude of the result grid (midpoint).
* **dyi** shape (192, 1) size of the result grid

* **zi** shape (29, 1) height of the result grid.
* **sepz** shape (29, 1) size of the result grid (midpoint).

* **exist** shape (2, 196 * 192 * 29) Those two list are in direct location
with the xi, yi, zi grid. The order is such that it cycle through x first, then
y, then z.

The first list contains 1 if the cell exists (is not in the air) and 0 else.
The second list gives the index of the column in F_land that corresponds to the
cell.

* **ind** shape (3, $n_points$) For each point in F_land, this contains the
xi-yi-zi index of the corresponding point.
Order: z -> y -> x.


#### DSM
10x10 m resolution

* **x** shape (1090, 1)
* **y** shape (1075, 1)
* **z** shape (1090, 1075)

#### Finer DSM
1x1 m resolution

* **x_fine** shape (5096, 1)
* **y_fine** shape (1, 4679)
* **z_fine** shape(5096, 4679)
