#!/usr/bin/env python
# coding: utf-8

# # Dynamics of Arc-continent collision - 3D

# In[1]:


import UWGeodynamics as GEO
from UWGeodynamics import visualisation as vis
from UWGeodynamics import dimensionalise
from UWGeodynamics import non_dimensionalise as nd
from underworld import function as fn
u = GEO.UnitRegistry
import math
import numpy as np
import os
import scipy


# In[2]:


gravity = 9.8 * u.meter / u.second**2
Tsurf   = 273.15 * u.degK
Tint    = 1573.0 * u.degK

kappa   = 1e-6   * u.meter**2 / u.second 

boxLength = 6000.0 * u.kilometer
boxHeight =  800.0 * u.kilometer
boxWidth  = 3000.0 * u.kilometer
dRho =   80. * u.kilogram / u.meter**3 # matprop.ref_density

use_scaling = False

# lithostatic pressure for mass-time-length
ref_stress = dRho * gravity * boxHeight
# viscosity of upper mante for mass-time-length
ref_viscosity = 1e20 * u.pascal * u.seconds

ref_time        = ref_viscosity/ref_stress
ref_length      = boxHeight
ref_mass        = (ref_viscosity*ref_length*ref_time)
#ref_temperature = Tint - Tsurf

KL = ref_length       
KM = ref_mass         
Kt = ref_time
#KT = ref_temperature

if use_scaling:
    # Disable internal scaling when using relrho_geo_material_properties.py
    KL = 1. * u.meter       
    KM = 1. * u.kilogram        
    Kt = 1. * u.second
    KT = 1. * u.degK

scaling_coefficients = GEO.get_coefficients()

scaling_coefficients["[length]"] = KL#.to_base_units()
scaling_coefficients["[time]"]   = Kt#.to_base_units()
scaling_coefficients["[mass]"]   = KM#.to_base_units()
#scaling_coefficients["[temperature]"]   = KT.to_base_units()

if use_scaling:
    import relrho_geo_material_properties as matprop #Material properties
else:
    import absrho_geo_material_properties as matprop #Material properties


# In[3]:


# shortcuts for parallel wrappers
barrier = GEO.uw.mpi.barrier
rank    = GEO.rank


# In[4]:



angle = 20
#nEls = (52,52)
shifted = 450 #distance of trench to ribbon
#nEls=(256,96,96)
#nEls=(30,30,30)
# nEls = (256,96,96)
#nEls = (96,48,48)
nEls = (256, 128)
#nEls = (12, 8,8)
dim = len(nEls)

outputPath = "Arc-Continent_Collision_3D_ReferenceModel"


# In[5]:


outputPath = os.path.join(os.path.abspath("."), outputPath)
if rank==0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
barrier()


# In[6]:


#Model Dimensions
boxLength = 6000.0 * u.kilometer
boxHeight =  800.0 * u.kilometer
boxWidth  = 3000.0 * u.kilometer
# Define our vertical unit vector using a python tuple
g_mag = 9.8 * u.meter / u.second**2

if dim == 2:
    minCoord   = (0., -boxHeight)
    maxCoord   = (boxLength, 0.)
    g_vec      = ( 0.0, -1.0 * g_mag )
    
else:
    minCoord   = (0., -boxHeight, 0.)
    maxCoord   = (boxLength, 0., boxWidth)
    g_vec      = ( 0.0, -1.0 * g_mag , 0.0 )
    
Model = GEO.Model(elementRes = nEls,
                  minCoord   = minCoord,
                  maxCoord   = maxCoord,
                  gravity    = g_vec,
                  outputDir  = outputPath)


# In[7]:


#Maximum and minimum viscosity - dimmensional and non-dimensional choices
if use_scaling:
    Model.defaultStrainRate = 1e-18 / u.second
    Model.minViscosity = 1e-1 * u.Pa * u.sec
    Model.maxViscosity = 1e5  * u.Pa * u.sec
else:
    Model.defaultStrainRate = 1e-18 / u.second
    Model.minViscosity = 1e19 * u.Pa * u.sec
    Model.maxViscosity = 1e25  * u.Pa * u.sec


# In[8]:


resolution = [ abs(Model.maxCoord[d]-Model.minCoord[d])/Model.elementRes[d] for d in range(Model.mesh.dim) ]
if rank == 0:
    print("Model resolution:")
    [ print(f'{d:.2f}') for d in resolution ]


# In[9]:


# Parameters for the inital material layout

# I assume here the origin is a the top, front, middle
# 'middle' being the slab hinge at top, front

slab_xStart = 2500. * u.kilometer
slab_dx = 3000.0 * u.kilometer  # was 7000 km in Moresi 2014
slab_dy =  100.0 * u.kilometer
slab_dz = 3000.0 * u.kilometer # this is the entire domain width
slab_layers = 4

slab_crust = 7.0 * u.kilometer


backarc_dx = 1200. * u.kilometer
backarc_dy =  100. * u.kilometer
backarc_xStart = slab_xStart - backarc_dx
backarc_layers = 2

trans_dx =  350. * u.kilometer
trans_dy =  100. * u.kilometer
trans_xStart = slab_xStart - backarc_dx - trans_dx
trans_layers = 2

craton_dx = 750. * u.kilometer
craton_dy = 150. * u.kilometer
craton_xStart = slab_xStart - backarc_dx - trans_dx - craton_dx
craton_layers = 2

#ribbon_dx =  500. * u.kilometer
#ribbon_dy =   50. * u.kilometer
#ribbon_dz = 1500. * u.kilometer 
#ribbon_xStart = slab_xStart + shifted * u.kilometer

bouyStrip_dx = 500. * u.kilometer
bouyStrip_dy =  50. * u.kilometer
bouyStrip_xStart = slab_xStart + slab_dx - bouyStrip_dx


# In[10]:


#variables for initialisation of shapes

s_y1 = -0*slab_dy # get dimensionality with slab_dy
s_y2 = -1*slab_dy/slab_layers
s_y3 = -2*slab_dy/slab_layers
s_y4 = -3*slab_dy/slab_layers

backarc_dx = 1200. * u.kilometer
backarc_dy =  100. * u.kilometer
backarc_xStart = slab_xStart - backarc_dx
backarc_layers = 2
dpert = 200 * u.km #dimensionalise(pert, u.km)

backarc_y1 = -0.*nd(backarc_dy)/backarc_layers
backarc_y2 = -1.*nd(backarc_dy)/backarc_layers

trans_y1 = -0.*nd(trans_dy)
trans_y2 = -1.*nd(trans_dy)

crat_y1 = -0.*nd(craton_dy)
crat_y2 = -1.*nd(craton_dy)


# In[11]:


# general shape functions
def slabGeo(x, y, dx, dy, BoxLength, orientation):
    if orientation==1:
        shape = [ (x,y), (x+dx,y), (x+dx,y-dy), (x,y-dy), (x-dpert,y-dy-dpert), (x-dpert,y-dpert) ]
    else:
        x=BoxLength-x
        dx=dx*orientation
        shape = [ (x,y), (x+dx,y), (x+(dx),y-dy), (x,y-dy), (x+dpert,y-dy-dpert), (x+dpert,y-dpert) ]
        
    return GEO.shapes.Polygon(shape)

def backarcGeo(x, y, dx, dy, BoxLength, orientation):
    if orientation==1:
        shape = [ (x,y), (x+dx,y), (x+dx-dy,y-dy), (x,y-dy)]
    else:
        x=BoxLength-x
        dx=dx*orientation
        shape = [ (x,y), (x+dx,y), (x+(dx)+dy,y-dy), (x,y-dy)]
    return GEO.shapes.Polygon(shape)


# ## Material Field

# ### Model Orientation

# In[12]:


orientation=-1


# In[13]:


#initialising all features as shapes
BoxLength=boxLength
# define coordinate uw.functions
fn_x = GEO.shapes.fn.input()[0]
fn_y = GEO.shapes.fn.input()[1]
fn_z = GEO.shapes.fn.input()[2]

UMantle=Model.add_material(name="UpperMantle", shape=GEO.shapes.Layer(top=0.*u.kilometer, bottom=-660.*u.kilometer))

op1 = slabGeo(slab_xStart, s_y1, slab_dx, slab_dy/slab_layers,BoxLength, orientation)
op1_fin = Model.add_material(name="oceanic plate 1", shape=op1)

op2 = slabGeo(slab_xStart, s_y2, slab_dx, slab_dy/slab_layers,BoxLength, orientation)
op2_fin = Model.add_material(name = "oceanic plate 2", shape=op2)

op3 = slabGeo(slab_xStart, s_y3, slab_dx, slab_dy/slab_layers,BoxLength, orientation)
op3_fin = Model.add_material(name = "oceanic plate 3", shape=op3)

op4 = slabGeo(slab_xStart, s_y4, slab_dx, slab_dy/slab_layers,BoxLength, orientation)
op4_fin = Model.add_material(name = "oceanic plate 4", shape=op4)

ba1 = backarcGeo(backarc_xStart, 0.*u.km, backarc_dx, 50.*u.km,BoxLength, orientation)
ba1_fin = Model.add_material(name="backArc1", shape=ba1)

ba2 = backarcGeo(backarc_xStart, -50.*u.km, backarc_dx-50*u.km, 50.*u.km,BoxLength, orientation)
ba2_fin = Model.add_material(name="backArc2", shape=ba2)

if orientation==-1:
    trans_xStart=BoxLength-trans_xStart
    trans_dx=trans_dx*orientation
    craton_xStart=BoxLength-craton_xStart
    craton_dx=craton_dx*orientation
    bouyStrip_xStart=BoxLength-bouyStrip_xStart
    bouyStrip_dx=bouyStrip_dx*orientation
    
#Shapes transtional crust and Craton
t1=GEO.shapes.Polygon(vertices=[(nd(trans_xStart), 0.),
                             (nd(trans_xStart+trans_dx), 0.),
                                
                             (nd(trans_xStart+trans_dx), nd(-trans_dy/trans_layers)),
                             (nd(trans_xStart), nd(-trans_dy/trans_layers))])

t2=GEO.shapes.Polygon(vertices=[(nd(trans_xStart), nd(-trans_dy/trans_layers)),
                             (nd(trans_xStart+trans_dx), nd(-trans_dy/trans_layers)),
                                
                             (nd(trans_xStart+trans_dx), nd(-trans_dy)),
                             (nd(trans_xStart), nd(-trans_dy))])

c1=GEO.shapes.Polygon(vertices=[(nd(craton_xStart), 0.),
                             (nd(craton_xStart+craton_dx), 0.),
                                
                             (nd(craton_xStart+craton_dx), nd(-craton_dy/craton_layers)),
                             (nd(craton_xStart), nd(-craton_dy/craton_layers))])

c2=GEO.shapes.Polygon(vertices=[(nd(craton_xStart), nd(-craton_dy/craton_layers)),
                             (nd(craton_xStart+craton_dx), nd(-craton_dy/craton_layers)),
                                
                             (nd(craton_xStart+craton_dx), nd(-craton_dy)),
                             (nd(craton_xStart), nd(-craton_dy))])

#Add shapes to model
    
# t1 = GEO.shapes.Box(top=Model.top,      bottom=-trans_dy/trans_layers, 
#                     minX=trans_xStart, maxX=trans_xStart+trans_dx)
t1_fin = Model.add_material(name="trans1", shape=t1)


# t2 = GEO.shapes.Box(top=-trans_dy/trans_layers, bottom=-trans_dy, 
#                     minX=trans_xStart, maxX=trans_xStart+trans_dx)
t2_fin = Model.add_material(name="trans2", shape=t2)


# c1 = GEO.shapes.Box(top=Model.top,      bottom=-craton_dy/craton_layers, 
#                     minX=craton_xStart, maxX=craton_xStart+craton_dx)
c1_fin = Model.add_material(name="craton1", shape=c1)


# c2 = GEO.shapes.Box(top=-craton_dy/craton_layers, bottom=-craton_dy,
#                     minX=craton_xStart, maxX=craton_xStart+craton_dx)
c2_fin = Model.add_material(name="craton2", shape=c2)


bs = GEO.shapes.Polygon(vertices=[(nd(bouyStrip_xStart), 0.),
                             (nd(bouyStrip_xStart+bouyStrip_dx), 0.),
                             (nd(bouyStrip_xStart+bouyStrip_dx), 0.-nd(bouyStrip_dy)),
                             (nd(bouyStrip_xStart), 0.-nd(bouyStrip_dy))])
bs_fin = Model.add_material(name="buoyStrip", shape=bs)


# In[14]:


ribbon_dx =  500. * u.kilometer
ribbon_dy =   50. * u.kilometer
ribbon_dz = 1500. * u.kilometer 
ribbon_xStart = slab_xStart + shifted * u.kilometer

if orientation==-1:
    ribbon_dx=ribbon_dx*orientation
    ribbon_xStart=BoxLength-ribbon_xStart
# always make 2D ribbon shape, if dim == 3 it's overwritten
#rib_shape = GEO.shapes.Box(top=0*u.km, bottom=-50*u.km,
#                           minX=ribbon_xStart, maxX=ribbon_xStart+ribbon_dx)

if dim == 3:
    #angle we want the ribbon rotated, can be +ve or -ve
    angle=20 #degres
    rad = np.radians(angle)

    #calculated associated half space normals
    nx = -np.cos(rad)
    nz = np.sin(rad)
    
    #Ribbon-Layer 1
    #floor
    hsp1 = GEO.shapes.HalfSpace(normal=(0.,-1.,0.), origin=(0, -25*u.km, 0.))
    #front
    hsp2 = GEO.shapes.HalfSpace(normal=(nx, 0, nz), origin=(ribbon_xStart+ribbon_dx,0.,Model.maxCoord[2]))
    #back
    hsp3 = GEO.shapes.HalfSpace(normal=(-nx, 0, -nz), origin=(ribbon_xStart,0.,Model.maxCoord[2]))
    #width
    hsp4 = GEO.shapes.HalfSpace(normal=(0, 0, -1), origin=(0.,0.,ribbon_dz))
    #top
    hsp5 = GEO.shapes.HalfSpace(normal=(0.,1.,0.), origin=(0, 0*u.km, 0.))

    rib_shape1 = hsp1&hsp2&hsp3&hsp4&hsp5
    
    #Ribbon-Layer 2
    #floor
    hsp11 = GEO.shapes.HalfSpace(normal=(0.,-1.,0.), origin=(0, -50*u.km, 0.))
    #front
    hsp21 = GEO.shapes.HalfSpace(normal=(nx, 0, nz), origin=(ribbon_xStart+ribbon_dx,0.,Model.maxCoord[2]))
    #back
    hsp31 = GEO.shapes.HalfSpace(normal=(-nx, 0, -nz), origin=(ribbon_xStart,0.,Model.maxCoord[2]))
    #width
    hsp41 = GEO.shapes.HalfSpace(normal=(0, 0, -1), origin=(0.,0.,ribbon_dz))
    #top
    hsp51 = GEO.shapes.HalfSpace(normal=(0.,1.,0.), origin=(0, -25*u.km, 0.))
    
    rib_shape2 = hsp11&hsp21&hsp31&hsp41&hsp51
    
    
#rib1 = Model.add_material(name="ribbon_1", shape=rib_shape1)

#rib2 = Model.add_material(name="ribbon_2", shape=rib_shape2)

rib1 = Model.add_material(name="ribbon_1")
rib2 = Model.add_material(name="ribbon_2")

op_change = Model.add_material(name="oceanic plate 1 after phase change")

#lm = Model.add_material(name="lower mantle", shape= fn_y < nd(-660.0 * 10**3 * u.meter))
lm=Model.add_material(name="lower mantle", shape=GEO.shapes.Layer(top=-660.*u.kilometer, bottom=Model.bottom))

added_material_list = [lm, op1_fin, op2_fin, op3_fin, op4_fin, ba1_fin, ba2_fin, t1_fin, 
                       t2_fin, c1_fin, c2_fin, bs_fin, op_change, rib1, rib2]


# In[15]:


figsize=(1000,300)
camera = ['zoom 100']#['rotate x 30']
boundingBox=( minCoord, maxCoord )

materialFilter = Model.materialField > 0


# ERROR with boundaringBox, maybe BUG for Okaluza
# figSwarm = vis.Figure(figsize=figsize, boundingBox=boundingBox )

# swarmPlot = vis.objects.Points(swarm, materialIndex, materialFilter, colours='gray', opacity=0.5, fn_size=2., 
#                                     discrete=True, colourBar=False, )

Fig = vis.Figure(figsize=(1200,400))

# Show single colour
# Fig.Points(Model.swarm, colour='gray', opacity=0.5, discrete=True, 
#            fn_mask=materialFilter, fn_size=2.0, colourBar=False)

# Show all glory
Fig.Points(Model.swarm, fn_colour=Model.materialField, 
           fn_mask=materialFilter, opacity=0.5, fn_size=2.0)

# # Save image to disk
# Fig.save("foobar.png")

# Rotate camera angle
#Fig.script(camera)

# Render in notebook
Fig.show()

#Fig.window()


# In[16]:


#assigning properties (density, viscosity, etc) to shapes.
# N.B. the default material 'Model' is assigned the 'upper mantle' properties

for i in Model.materials:
    for j in matprop.material_list:
        if i.name == j["name"]:
            if rank == 0: print(i.name)
            i.density = j["density"]
            i.viscosity = j["viscosity"]
            c0 = j["cohesion"] if j.get('cohesion') else None
            c1 = j["cohesion2"] if j.get('cohesion2') else c0
            if c0 is not None:
                i.plasticity = GEO.VonMises(cohesion = c0, cohesionAfterSoftening = c1)
                                       # TODO epsilon1=0., epsilon2=0.1

if rank == 0: print("Assigning material properties...")

#print("properties")
# **Eclogite transition**
# 
# Assume that the oceanic crust transforms instantaneously and completely to eclogite at a depth of 150 km


# In[17]:


op1_fin.phase_changes = GEO.PhaseChange((Model.y < nd(-150.*u.kilometers)),
                                          op_change.index)
# Not sure about the others
# op2_fin.phase_changes = GEO.PhaseChange((Model.y < nd(-150.*u.kilometers)),
#                                           op_change.index)
# op3_fin.phase_changes = GEO.PhaseChange((Model.y < nd(-150.*u.kilometers)),
#                                           op_change.index)
# op4_fin.phase_changes = GEO.PhaseChange((Model.y < nd(-150.*u.kilometers)),
#                                           op_change.index)

store = vis.Store("store" + str(shifted))
figure_one = vis.Figure(store, figsize=(1200,400))
figure_one.append(Fig.Points(Model.swarm, fn_colour=Model.materialField, fn_mask=materialFilter, opacity=0.5, fn_size=2.0))
store.step = 0
figure_one.save()


# In[18]:


figsize=(1000,300)
camera = ['rotate x 30']
boundingBox=( minCoord, maxCoord )

materialFilter = Model.materialField > 0


# ERROR with boundaringBox, maybe BUG for Okaluza
# figSwarm = vis.Figure(figsize=figsize, boundingBox=boundingBox )

# swarmPlot = vis.objects.Points(swarm, materialIndex, materialFilter, colours='gray', opacity=0.5, fn_size=2., 
#                                     discrete=True, colourBar=False, )

Fig = vis.Figure(figsize=(1200,400))

# Show single colour
# Fig.Points(Model.swarm, colour='gray', opacity=0.5, discrete=True, 
#            fn_mask=materialFilter, fn_size=2.0, colourBar=False)

# Show all glory
Fig.Points(Model.swarm, fn_colour=Model.materialField, fn_mask=materialFilter,colours='dem1', opacity=0.5, fn_size=2.0)

# Save image to disk
# Fig.save("Figure_1.png")

# Render in notebook
Fig.show()
#Fig.window()


# In[19]:


Fig = vis.Figure(figsize=(1200,400))
Fig.Points(Model.swarm, fn_colour=Model.materialField, colours='dem1', fn_size=1.0)
Fig.show()


# ## Passive Tracers

# In[20]:


#Interpolate Tracers
def interpolateTracer(coord1,coord2,nPoints):
    x=np.linspace(coord1[0],coord2[0],nPoints)#*u.kilometer
    m=(coord2[1]-coord1[1])/(coord2[0]-coord1[0])
    y=m*(x-coord1[0])+coord1[1]
    return x,y,m

def getMin(a,b):
    aux=0
    if a<b:
        aux=a
    else:
        aux=b
    return aux
    
def getMax(a,b):
    aux=0
    if a>b:
        aux=a
    else:
        aux=b
    return aux

# def filterSurface(Xs,Ys):
    
#     for x,y in zip (Xs,Ys):
    
#     return
def build_tracer_swarm_craton(name, minX, maxX, numX, y, minZ, maxZ, numZ,BoxLength, orientation):
    # wrapper for `Model.add_passive_tracers()` which doesn't take dimensional values

    minX = GEO.nd(minX) ; maxX = GEO.nd(maxX)
    minZ = GEO.nd(minZ) ; maxZ = GEO.nd(maxZ)
    
    xx = np.linspace(minX, maxX, abs(numX))
    yy = np.array([GEO.nd(y)]) ##change!! for accounting the angle.. function of XX values
    zz = np.linspace(minZ, maxZ, numZ)

    xx, yy, zz = np.meshgrid(xx,yy,zz)
    
    coords = np.ndarray((xx.size, 3))
    coords[:,0] = xx.ravel()
    coords[:,1] = yy.ravel()
    coords[:,2] = zz.ravel()
    tracers = Model.add_passive_tracers(name, vertices = coords)
    return tracers

def build_tracer_swarm_backaArc(name, minX, maxX, numX, y, minZ, maxZ, numZ,BoxLength, orientation):
    # wrapper for `Model.add_passive_tracers()` which doesn't take dimensional values
    if orientation==1:
        minX = GEO.nd(minX) ; maxX = GEO.nd(maxX)
        minZ = GEO.nd(minZ) ; maxZ = GEO.nd(maxZ)
    else:
        minX = GEO.nd(BoxLength-minX) ; maxX = GEO.nd(BoxLength-maxX)
        minZ = GEO.nd(minZ) ; maxZ = GEO.nd(maxZ)
    
    xx = np.linspace(minX, maxX, abs(numX))
    yy = np.array([GEO.nd(y)]) ##change!! for accounting the angle.. function of XX values
    zz = np.linspace(minZ, maxZ, numZ)

    xx, yy, zz = np.meshgrid(xx,yy,zz)
    
    coords = np.ndarray((xx.size, 3))
    coords[:,0] = xx.ravel()
    coords[:,1] = yy.ravel()
    coords[:,2] = zz.ravel()
    tracers = Model.add_passive_tracers(name, vertices = coords)
    return tracers

def build_tracer_slab_swarm(name, minX, maxX, numX, Y, minZ, maxZ, numZ,dpert,BoxLength, orientation):
    # wrapper for `Model.add_passive_tracers()` which doesn't take dimensional values
    if orientation==1:
        minX = GEO.nd(minX) ; maxX = GEO.nd(maxX)
        minZ = GEO.nd(minZ) ; maxZ = GEO.nd(maxZ)
    else:
        minX = GEO.nd(BoxLength-minX) ; maxX = GEO.nd(BoxLength-maxX)
        minZ = GEO.nd(minZ) ; maxZ = GEO.nd(maxZ)
        dpertY=-dpert
    
    #Flat segment
    xx = np.linspace(minX, maxX, abs(numX))
    yy = np.array([GEO.nd(Y)]) ##change!! for accounting the angle.. function of XX values
    zz = np.linspace(minZ, maxZ, numZ)

    xx, yy, zz = np.meshgrid(xx,yy,zz)
    
    coords = np.ndarray((xx.size, 3))
    coords[:,0] = xx.ravel()
    coords[:,1] = yy.ravel()
    coords[:,2] = zz.ravel()
    
    tracers = Model.add_passive_tracers(name, vertices = coords)
    
    return tracers


# In[21]:


# (x-dpert,y-dy-dpert)
# (x+dx,y)


# In[22]:


def slabGeoTest(x, y, dx, dy, BoxLength, orientation):
    if orientation==1:
        shape = [ (x,y), (x+dx,y), (x+dx,y-dy), (x,y-dy), (x-dpert,y-dy-dpert), (x-dpert,y-dpert) ]
    else:
        x=BoxLength-x
        dx=dx*orientation
        shape = [ (x,y), (x+dx,y), (x+(dx),y-dy), (x,y-dy), (x+dpert,y-dy-dpert), (x+dpert,y-dpert) ]
        
    return GEO.shapes.Polygon(shape)


# ### Slab-Tracers

# In[23]:
#print("tracer11")

# if orientation==-1:
#     slab_xStart=BoxLength-slab_xStart
#     slab_dx=slab_dx*orientation

if dim == 3:
    #layer 1
    sp1 = build_tracer_slab_swarm("slab_l1",
                                 slab_xStart , slab_xStart+slab_dx, int(np.ceil(slab_dx/resolution[0])),
                                 0, Model.minCoord[2]+resolution[2]/2,Model.maxCoord[2]-resolution[2]/2, Model.elementRes[2]-1,dpert,BoxLength, orientation)
    
    y = -25*u.km
    sp2 = build_tracer_slab_swarm("slab_l2",
                                 slab_xStart , slab_xStart+slab_dx, int(np.ceil(slab_dx/resolution[0])),
                                 y, Model.minCoord[2]+resolution[2]/2,Model.maxCoord[2]-resolution[2]/2, Model.elementRes[2]-1,dpert,BoxLength, orientation)
    
    y = -50*u.km
    sp3 = build_tracer_slab_swarm("slab_l3",
                                 slab_xStart , slab_xStart+slab_dx, int(np.ceil(slab_dx/resolution[0])),
                                 y, Model.minCoord[2]+resolution[2]/2,Model.maxCoord[2]-resolution[2]/2, Model.elementRes[2]-1,dpert,BoxLength, orientation)
    
    y = -75*u.km
    sp4 = build_tracer_slab_swarm("slab_l4",
                                 slab_xStart , slab_xStart+slab_dx, int(np.ceil(slab_dx/resolution[0])),
                                 y, Model.minCoord[2]+resolution[2]/2,Model.maxCoord[2]-resolution[2]/2, Model.elementRes[2]-1,dpert,BoxLength, orientation)
    
    
    
    sp1.add_tracked_field(Model.strainRate, "sr_tensor", units=u.sec**-1, dataType="double", count=6)
    sp2.add_tracked_field(Model.strainRate, "sr_tensorb", units=u.sec**-1, dataType="double", count=6)
    sp3.add_tracked_field(Model.strainRate, "sr_tensorc", units=u.sec**-1, dataType="double", count=6)
    sp4.add_tracked_field(Model.strainRate, "sr_tensord", units=u.sec**-1, dataType="double", count=6)
    
    sp1.add_tracked_field(Model.velocityField, "velocity1", units=u.centimeter/ u.year, dataType="double", count=3)
    sp2.add_tracked_field(Model.velocityField, "velocity2", units=u.centimeter/ u.year, dataType="double", count=3)
    sp3.add_tracked_field(Model.velocityField, "velocity3", units=u.centimeter/ u.year, dataType="double", count=3)
    sp4.add_tracked_field(Model.velocityField, "velocity4", units=u.centimeter/ u.year, dataType="double", count=3)
#     tracersDip = build_tracer_slab_swarm("slab_l1",
#                                  slab_xStart , slab_xStart+slab_dx, int(np.ceil(slab_dx/resolution[0])),
#                                  0, 
#                                  Model.minCoord[2]+resolution[2]/2,Model.maxCoord[2]-resolution[2]/2, Model.elementRes[2]-1,dpert,BoxLength, orientation)[1]
#     tracersDip.add_tracked_field(Model.strainRate, "sr_tensor", units=u.sec**-1, dataType="double", count=6)
    


# ### Back-arc tracers

# In[24]:



# # build 2 tracer swarms, one on the surface, and one 25 km down 

if dim == 3:
    baT1 = build_tracer_swarm_backaArc("ba_surface",
                                 backarc_xStart, backarc_xStart+backarc_dx, int(np.ceil(backarc_dx/resolution[0])),
                                 0, 
                                 Model.minCoord[2]+resolution[2]/2,Model.maxCoord[2]-resolution[2]/2, Model.elementRes[2]-1,BoxLength, orientation)
    

#     2nd tracers must be called something different to the 1st, i.e. 'tracers'
    y = -15*u.km
    baT2 = build_tracer_swarm_backaArc("ba_subsurf",
                                 backarc_xStart, backarc_xStart+backarc_dx+y, int(np.ceil(backarc_dx/resolution[0])),
                                 y, 
                                 Model.minCoord[2]+resolution[2]/2,Model.maxCoord[2]-resolution[2]/2, Model.elementRes[2]-1,BoxLength, orientation)
    
    baT1.add_tracked_field(Model.strainRate, "sr_tensor", units=u.sec**-1, dataType="double", count=6)
    baT2.add_tracked_field(Model.strainRate, "sr_tensorb", units=u.sec**-1, dataType="double", count=6)# # build 2 tracer swarms, one on the surface, and one 25 km down
    
    baT1.add_tracked_field(Model.projStressTensor, "s_tensor", units=u.megapascal, dataType="double", count=6)
    baT2.add_tracked_field(Model.projStressTensor, "s_tensorb", units=u.megapascal, dataType="double", count=6)


# ### Craton Tracers

# In[25]:


if dim == 3:
    cratT1 = build_tracer_swarm_craton("cra_surface",
                                 craton_xStart, craton_xStart+craton_dx, int(np.ceil(craton_dx/resolution[0])),
                                 0, 
                                 Model.minCoord[2]+resolution[2]/2,Model.maxCoord[2]-resolution[2]/2, Model.elementRes[2]-1,BoxLength, orientation)
    

#     2nd tracers must be called something different to the 1st, i.e. 'tracers'
    y = -15*u.km
    cratT2 = build_tracer_swarm_craton("cra_subsurf",
                                 craton_xStart, craton_xStart+craton_dx+y, int(np.ceil(craton_dx/resolution[0])),
                                 y, 
                                 Model.minCoord[2]+resolution[2]/2,Model.maxCoord[2]-resolution[2]/2, Model.elementRes[2]-1,BoxLength, orientation)
    
    
    cratT1.add_tracked_field(Model.strainRate, "sr_tensor", units=u.sec**-1, dataType="double", count=6)
    cratT2.add_tracked_field(Model.strainRate, "sr_tensorb", units=u.sec**-1, dataType="double", count=6)# # build 2 tracer swarms, one on the surface, and one 25 km down

    cratT1.add_tracked_field(Model.velocityField, "velocity", units=u.centimeter/ u.year, dataType="double", count=3)
    cratT2.add_tracked_field(Model.velocityField, "velocityb", units=u.centimeter/ u.year, dataType="double", count=3)


# In[26]:


FigTracers = vis.Figure(figsize=(1200,400))

# Show single colour
# Fig.Points(Model.swarm, colour='gray', opacity=0.5, discrete=True, 
#            fn_mask=materialFilter, fn_size=2.0, colourBar=False)


def get_show_tracer(name, colours):
    t = Model.passive_tracers.get(name)
    if not t: raise RuntimeError("ERROR: fine tracer called ", name)
    
    t.variables[0] #is the cell index, not of interest, only used to display in 'store'
    FigTracers.Points(t, t.variables[0],fn_size=2., colours=colours,opacity=0.5,colourBar=False)

if dim == 3:

    get_show_tracer(name="ba_surface", colours="#22BBBB")
    get_show_tracer(name='ba_subsurf', colours="#335588")
    #get_show_tracer(name='slab_l1', colours="#335588")
    get_show_tracer(name='slab_l1', colours="#335588")
    get_show_tracer(name='slab_l2', colours="#335588")
    get_show_tracer(name='slab_l3', colours="#335588")
    get_show_tracer(name='slab_l4', colours="#335588")
    get_show_tracer(name='cra_surface', colours="#335588")
    get_show_tracer(name='cra_subsurf', colours="#335588")
#         get_show_tracer(name='slab', colours="Gray40 Goldenrod")
##         get_show_tracer(name='cont', colours="#335588 #22BBBB")
##         get_show_tracer(name='arc', colours="Goldenrod Grey41")
#         get_show_tracer(name='buoy', colours="#335588 #335588")
#
## def output_tracers(i):
##
##     store.step = i
##     # Rotate camera angle
##     #FigTracers.script(camera)
##
##     #FigTracers.save('foobar.png')
#
## Render in notebook
## Show all glory
#FigTracers.Points(Model.swarm, fn_colour=Model.materialField, fn_mask=materialFilter, opacity=0.5, fn_size=2.0)
#
#FigTracers.show()
#FigTracers.window()


# ## Viscosity

# In[27]:


#VIscosity limits
Model.minViscosity = 1e19 * u.Pa * u.sec
Model.maxViscosity = 1e25 * u.Pa * u.sec

#MinViscosity for materials
op1_fin.minViscosity =10**(20) * u.pascal * u.second
op2_fin.minViscosity = 10**(20) * u.pascal * u.second
op3_fin.minViscosity = 10**(20) * u.pascal * u.second
op4_fin.minViscosity = 10**(20) * u.pascal * u.second
ba1_fin.minViscosity= 10**(20) * u.pascal * u.second
ba2_fin.minViscosity=10**(20) * u.pascal * u.second
t1_fin.minViscosity = 10**(20) * u.pascal * u.second
t2_fin.minViscosity = 10**(20) * u.pascal * u.second
c1_fin.minViscosity = 10**(20) * u.pascal * u.second
c2_fin.minViscosity = 10**(20) * u.pascal * u.second
bs_fin.minViscosity = 10**(20) * u.pascal * u.second
rib1.minViscosity = 10**(19) * u.pascal * u.second
rib2.minViscosity = 10**(19) * u.pascal * u.second


# In[28]:


#Velocity BCs
if dim == 2:
    Model.set_velocityBCs(left=[0., None],
                     right=[0.,None],
                     bottom=[0., 0.],
                     top=[None, 0.])
else:
    Model.set_velocityBCs( left=[0.,None,None], right=[0.,None,None],
                       front=[None,0.,None], back=[None,0.,None],
                       bottom=[None,None,0.], top=[None,None,0.])

#print("BCs")
# In[29]:


Fig = vis.Figure(figsize=(1200,400))

# Show all glory
Fig.Points(Model.swarm, fn_colour=GEO.dimensionalise(Model.densityField,u.kilogram / u.metre**3), fn_mask=materialFilter, opacity=0.5, fn_size=2.0)
#Fig.Points(Model.swarm, fn_colour=(Model.densityField), fn_mask=materialFilter, opacity=0.5, fn_size=2.0)
# Rotate camera angle
#Fig.script(camera)

# Render in notebook
#Fig.window()
Fig.show()


# In[30]:


Fig = vis.Figure(figsize=(1200,400))

# Show all glory
Fig.Points(Model.swarm, fn_colour=GEO.dimensionalise(Model.viscosityField,u.pascal * u.second), fn_mask=materialFilter, opacity=0.5, fn_size=2.0,logScale=True)
#Fig.Points(Model.swarm, fn_colour=(Model.densityField), fn_mask=materialFilter, opacity=0.5, fn_size=2.0)
# Rotate camera angle
#Fig.script(camera)

# Render in notebook
#Fig.window()
Fig.show()


# In[31]:


if rank == 0: print("Calling init_model()...")
Model.init_model()


# In[32]:


# force the Eclogite phase transition before the model begins
Model._phaseChangeFn()


# In[33]:


fout = outputPath+'/FrequentOutput.dat'
if rank == 0:
    with open(fout,'a') as f:
         f.write('#step\t time(Myr)\t Vrms(cm/yr)\n')
            
def post_solve_hook():
    vrms = Model.stokes_SLE.velocity_rms()
    step = Model.step
    time = Model.time.m_as(u.megayear)
    
#    if dim==3: output_tracers(step)
    
    if rank == 0:
        with open(fout,'a') as f:
             f.write(f"{step}\t{time:5e}\t{vrms:5e}\n")

        store.step += 1
        figure_one.save()
        figure_one.save("store" + str(shifted) + str(store.step))
        
    # DEBUG CODE
    #subMesh = Model.mesh.subMesh
    #jeta.data[:] = Model._viscosityFn.evaluate(subMesh)
    #jrho.data[:] = Model._densityFn.evaluate(subMesh)
    #jsig.data[:] = jeta.data[:] * Model.strainRate_2ndInvariant.evaluate(subMesh)
        
Model.post_solve_functions["Measurements"] = post_solve_hook


# In[34]:


#We can test different solvers by uncommentting this section
#solver = Model.solver
#scr_rtol = 1e-6
## Schur complement solver options
#solver.options.scr.ksp_rtol = scr_rtol
#solver.options.scr.ksp_type = "fgmres"
##solver.options.main.list()
#
## Inner solve (velocity), A11 options
#solver.options.A11.ksp_rtol = 1e-1 * scr_rtol
#solver.options.A11.ksp_type = "fgmres"
#solver.options.A11.list

##Testing Julians suggestion
#scr_rtol = 1e-6
### Schur complement solver options
#Model.solver.options.scr.ksp_rtol = scr_rtol
### Inner solve (velocity), A11 options
#Model.solver.options.A11.ksp_rtol = 1e-1 * scr_rtol


# In[35]:


if dim == 2:
    Model.solver.set_inner_method("lu")
    GEO.rcParams["initial.nonlinear.tolerance"] = 1e-1
    #Model.solver.options.scr.ksp_rtol = 1e-6 # small tolerance is good in 2D, not sure if too tight for 3D
#else:
#    solver.set_inner_method("superludist")
#    solver.options.A11.ksp_rtol = 1.0e-5
#    solver.options.scr.ksp_rtol = 1.0e-6 

# solver.print_petsc_options()

Fig = vis.Figure(figsize=(1200,400))
Fig.Points(Model.swarm, fn_colour=2.*Model.viscosityField*Model.strainRate_2ndInvariant, colours='dem1', logScale=True,fn_size=1.0)
# Fig.show()
Fig = vis.Figure(figsize=(1200,400))
Fig.Points(Model.swarm, fn_colour=Model._stressField, logScale=True, colours='dem1', fn_size=1.0)
#Fig.show()Fig = vis.Figure(figsize=(1200,400))
Fig.Points(Model.swarm, fn_colour=Model.materialField, colours='dem1', fn_size=1.0)
#Fig.show()
# In[36]:


## debugging code to generate initial fields for viscosity and density ##
# fields output used to analyse initial setup

#jeta = Model.add_submesh_field(name="cell_vis", nodeDofCount=1)
#jrho = Model.add_submesh_field(name="cell_rho", nodeDofCount=1)
#jsig = Model.add_submesh_field(name="cell_sig", nodeDofCount=1)
#
#GEO.rcParams["default.outputs"].append("cell_vis")
#GEO.rcParams["default.outputs"].append("cell_rho")
#GEO.rcParams["default.outputs"].append("cell_sig")
#
#subMesh = Model.mesh.subMesh
#jeta.data[:] = Model._viscosityFn.evaluate(subMesh)
#jrho.data[:] = Model._densityFn.evaluate(subMesh)
#jsig.data[:] = 2. * jeta.data[:] * Model.strainRate_2ndInvariant.evaluate(subMesh)


# ### Fields to save

# In[37]:


#Data to Save
outputss=['pressureField',
         'strainRateField',
         'velocityField',
          'projStressField',
          'projTimeField',
           'projMaterialField',
         'projViscosityField',
         'projStressField',
         'projMeltField',
          'projPlasticStrain',
         'projDensityField',
         'projStressTensor',
         ]
GEO.rcParams['default.outputs']=outputss


# In[ ]:


# make the rib shape, 
# best moved to where the other shapes are created above for consistency.

# if dim == 2:
#     rib_shape = GEO.shapes.Box(top=0*u.km, bottom=-50*u.km,
#                            minX=ribbon_xStart, maxX=ribbon_xStart+ribbon_dx)
    
# if dim == 3:
#     #angle we want the ribbon rotated, can be +ve or -ve
#     angle=20 #degres
#     rad = np.radians(angle)

#     #calculated associated half space normals
#     nx = -np.cos(rad)
#     nz = np.sin(rad)
    
#     #Ribbon-Layer 1
#     #floor
#     hsp1 = GEO.shapes.HalfSpace(normal=(0.,-1.,0.), origin=(0, -25*u.km, 0.))
#     #front
#     hsp2 = GEO.shapes.HalfSpace(normal=(nx, 0, nz), origin=(ribbon_xStart+ribbon_dx,0.,Model.maxCoord[2]))
#     #back
#     hsp3 = GEO.shapes.HalfSpace(normal=(-nx, 0, -nz), origin=(ribbon_xStart,0.,Model.maxCoord[2]))
#     #width
#     hsp4 = GEO.shapes.HalfSpace(normal=(0, 0, -1), origin=(0.,0.,ribbon_dz))
#     #top
#     hsp5 = GEO.shapes.HalfSpace(normal=(0.,1.,0.), origin=(0, 0*u.km, 0.))

#     rib_shape1 = hsp1&hsp2&hsp3&hsp4&hsp5
    
#     #Ribbon-Layer 2
#     #floor
#     hsp11 = GEO.shapes.HalfSpace(normal=(0.,-1.,0.), origin=(0, -50*u.km, 0.))
#     #front
#     hsp21 = GEO.shapes.HalfSpace(normal=(nx, 0, nz), origin=(ribbon_xStart+ribbon_dx,0.,Model.maxCoord[2]))
#     #back
#     hsp31 = GEO.shapes.HalfSpace(normal=(-nx, 0, -nz), origin=(ribbon_xStart,0.,Model.maxCoord[2]))
#     #width
#     hsp41 = GEO.shapes.HalfSpace(normal=(0, 0, -1), origin=(0.,0.,ribbon_dz))
#     #top
#     hsp51 = GEO.shapes.HalfSpace(normal=(0.,1.,0.), origin=(0, -25*u.km, 0.))
    
#     rib_shape2 = hsp11&hsp21&hsp31&hsp41&hsp51

# a single parameter to switch between restart workflow or not.
RESTART = False #first run should be false, then true
#GEO.rcParams["initial.nonlinear.tolerance"] = 4e-2
#print("check1")
if RESTART == False: 
    #Model.run_for(duration=10*u.megayear,checkpoint_interval=0.25*u.megayear) #here runs first time before onset of ribbon
    print("state")
    Model.run_for(nstep=20, checkpoint_interval=1)
    
else:
    # if the restartDir is the same as the current Model.outputDir we don't
    # require the `restartDir` argument. nstep MUST be 0 to allow the fancy
    # ribbon addition below - Dont forget this does not calcualte anything, just for putting the ribbon and re-staring
    Model.run_for(nstep=0, restartStep=80) #here i put where In time i want the ribbon, step is where the base model is
    
    # do fancy ribbon addition via shape
    matField = Model.swarm_variables['materialField']
    
    isInsideShape1 = rib_shape1.evaluate(Model.swarm.data)# where particle is inside the ribbon update it's materialField index
    
    Model.swarm_variables['materialField'].data[:] = np.where(isInsideShape1 == True, 
                                                              rib1.index, matField.data )
    isInsideShape2 = rib_shape2.evaluate(Model.swarm.data)# where particle is inside the ribbon update it's materialField index
    
    Model.swarm_variables['materialField'].data[:] = np.where(isInsideShape2 == True, 
                                                              rib2.index, matField.data )
    


# In[ ]:


# to visualise after workflow switch
Fig = vis.Figure(figsize=(1200,400))
Fig.Points(Model.swarm, fn_colour=Model.materialField, colours='dem1', fn_size=1.0)
Fig.show()

#print("hey")
# In[ ]:


#GEO.rcParams['nonlinear.max.iterations'] = 20
#GEO.rcParams['initial.nonlinear.max.iterations'] = 20
GEO.rcParams["initial.nonlinear.tolerance"] = 4e-2

#As the model was re-started before now it runs for the desired time!, no need to re-start again
# now continue running model as usual.
Model.run_for(duration=50*u.megayear,checkpoint_interval=0.25*u.megayear)
              #,restartStep=198)


# In[ ]:


figsize=(1000,300)
camera = ['rotate x 30']
boundingBox=( minCoord, maxCoord )

materialFilter = Model.materialField > 0
Fig = vis.Figure(figsize=(1200,400))


# In[ ]:


# Show single colour
# Fig.Points(Model.swarm, colour='gray', opacity=0.5, discrete=True, 
#            fn_mask=materialFilter, fn_size=2.0, colourBar=False)


# In[ ]:


# Save image to disk
# Fig.save("Figure_1.png")


# In[ ]:


# Show all glory
Fig.Points(Model.swarm, fn_colour=Model.materialField, 
           fn_mask=materialFilter, opacity=0.5, fn_size=2.0)


# In[ ]:


# Render in notebook
Fig.show()


# In[ ]:


from UWGeodynamics import visualisation as vis
view = vis.Viewer("store650")
steps = view.steps
view.step = steps[80]
for i in view.figures:
    view.figure(i)
    view["title"] = "Timestep ##"
    view.window()


# In[ ]:


figs = view.figures
steps = view.steps
view.step = steps[20]
for i in view.figures:
    view.figure(i)
    view["title"] = "Timestep ##"
    view.show()


# In[ ]:


figs = view.figures
steps = view.steps
view.step = steps[90]
for i in view.figures:
    view.figure(i)
    view["title"] = "Timestep ##"
    view.show()

