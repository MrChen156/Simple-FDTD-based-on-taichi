import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.vulkan)

# 定义一个一维时变电磁场，用一个field表示对应时刻下的这个场
# 场的大小：多少个空间步长
Space_size = 64
Space = range(Space_size)
# 模拟的时间步长
Time_steps = 300000
# 场的各项参数
imp0 = ti.field(ti.f32, shape=(Space_size,Space_size))
Cezh = 1.0/np.sqrt(2.0)
Ceze = 1.0
Chxh = 1.0
Chxe = 1.0/np.sqrt(2.0)
Chyh = 1.0
Chye = 1.0/np.sqrt(2.0)
# 定义电场
Electric_field = ti.field(ti.f32, shape=(Space_size,Space_size))
# 定义磁场，0-分量为x，1-分量为y
Magnetic_field = ti.Vector.field(n=2, dtype=ti.f32, shape=(Space_size,Space_size))
# 常用参数
Electric_max = 0

#电磁学常数生成
@ti.kernel
def Initialize_imp0():
    for i in imp0:
        imp0[i] = ti.sin(i/100)# ti.random(dtype=ti.f32)

# 定义电场和磁场更新函数
# 电场更新函数
@ti.func
def Electric_renew():
    for i in range(1,Space_size):
        for j in range(1,Space_size):
            Electric_field[i,j] = Electric_field[i,j]*Ceze + (Magnetic_field[i,j][1] - Magnetic_field[i-1,j][1] -\
                                                              Magnetic_field[i,j][0] + Magnetic_field[i,j-1][0])*Cezh
    # for i in range(1, Space_size):
    #     for j in range(1, Space_size):
    #         Electric_field[0,j] = Electric_field[1,j]
    #         Electric_field[i,0] = Electric_field[i,1]

# 磁场更新函数
@ti.func
def Magnetic_renew():
    for i in range(0,Space_size-1):
        for j in range(0,Space_size-1):
            Magnetic_field[i,j][0] = Magnetic_field[i,j][0]*Chxh + (Electric_field[i,j+1] - Electric_field[i,j])*Chxe
            Magnetic_field[i,j][1] = Magnetic_field[i,j][1]*Chyh + (Electric_field[i+1,j] - Electric_field[i,j])*Chye
    # for i in range(0, Space_size - 1):
    #     for j in range(0, Space_size - 1):
    #         Magnetic_field[Space_size - 1,j] = Magnetic_field[Space_size - 2,j]
    #         Magnetic_field[i,Space_size-1] = Magnetic_field[i,Space_size-2]

# 硬源
@ti.func
def Electric_source(t_step: ti.f32):#t_step是时间步
    for i, j in Electric_field:
        # Electric_field[i,j] = 0
        Electric_field[10,j] = 1-ti.sin(t_step/50)
        Electric_field[i,Space_size/2] = 1-ti.cos(t_step/50)

# 总更新函数
@ti.kernel
def Electric_magnetic_renew(time_step: ti.f32):
    Electric_source(t_step=time_step)
    Magnetic_renew()
    Electric_renew()

# 初始化磁场函数
@ti.kernel
def Initialize():
    for i, j in Magnetic_field:
        Magnetic_field[i,j][0] = 0
        Magnetic_field[i,j][1] = 0
    for i, j in Electric_field:
        Electric_field[i,j] = 0

# 生成空间参数
# Initialize_imp0()
# imp0_np = imp0.to_numpy()

# 生成GUI
gui = ti.GUI("FDTD", res=(Space_size, Space_size))

for time_step_num in range(Time_steps):
    Initialize()
    Electric_magnetic_renew(time_step=time_step_num)
    # Magnetic_output = Magnetic_field.to_numpy()
    Electric_output = Electric_field.to_numpy()
    Electric_max = np.max(np.abs(Electric_output))
    Electric_output = np.abs(Electric_output)/Electric_max
    #Electric_field = Electric_field.from_numpy(Electric_output)
    gui.set_image(Electric_output)
    gui.show()
