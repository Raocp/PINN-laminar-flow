# PINN-laminar-flow
Physics-informed neural network (PINN) for solving fluid dynamics problems

# Reference paper
This repo include the implementation of mixed-form physics-informed neural networks in paper: 

[Chengping Rao, Hao Sun and Yang Liu. Physics-informed deep learning for incompressible laminar flows.](https://arxiv.org/abs/2002.10558)

# Description for each folder
- **FluentReferenceMu002**: Reference solution from Ansys Fluent for steady flow;
<!--- - **FluentReferenceUnsteady**: Reference solution from Ansys Fluent for unsteady flow; --->
- **PINN_steady**: Implementation for steady flow with PINN;
- **PINN_unsteady**: Implementation for unsteady flow with PINN;

# Results overview

![](https://github.com/Raocp/PINN-laminar-flow/blob/master/PINN_steady/uvp.png)

> Steady flow past a cylinder (left: physics-informed neural network; right: Ansys Fluent.)


![](https://github.com/Raocp/PINN-laminar-flow/blob/master/PINN_unsteady/uvp_animation.gif)

> Transient flow past a cylinder (physics-informed neural network result)

# Note
- These implementations were developed and tested on the GPU version of TensorFlow 1.10.0. 
