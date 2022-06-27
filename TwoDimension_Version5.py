# 使用taichi模拟电磁波在二维空间中的传播
# 导入包
import taichi as ti
import numpy as np
# taichi激活
ti.init(arch=ti.vulkan)
# 常数设置区
# 当前只考虑均匀空间
c = 3e8  # 真空光速
pi = 3.1415926  # 圆周率
mu_0 = 4.0*pi*1.0e-7  # 真空磁导率
eps_0 = 8.854e-12  # 真空介电常数
Nx = 1400  # x轴向上的Yee元胞数量
Ny = 1400  # y轴向上的Yee元胞数量
Nt = 3000  # 最长时间步数
f_0 = 6e14  # 激励源频率
omega = 2.0*pi*f_0  # 激励源角频率
Dx = 2e-9  # (c/f_0)/2  # Yee元胞在x方向上的长度
Dy = 2e-9  # (c/f_0)/2  # Yee元胞在y方向上的长度
Sc = 0.5  # 稳定性因子
Dt = Sc / c * Dx  # 时间步长
# 变量区
# 电场值
Electric_field = ti.field(dtype=ti.f32, shape=(Nx, Ny))
Electric_field_old = ti.field(dtype=ti.f32, shape=(Nx, Ny))
# 磁场值
# 为了方便起见，用两个field分别储存x方向和y方向分量
Magnetic_field_x = ti.field(dtype=ti.f32, shape=(Nx, Ny))
Magnetic_field_y = ti.field(dtype=ti.f32, shape=(Nx, Ny))
# 激励源坐标
Source_x = int((Nx+1)/2)
Source_y = int((Ny+1)/2)
# 电场激励
pulse = 0.0
# 场更新系数
CHy = Dt/mu_0/Dx
CHx = Dt/mu_0/Dy
CEzHy = Dt/eps_0/Dx
CEzHx = Dt/eps_0/Dy

mu0_field = ti.field(dtype=ti.f32, shape=(Nx, Ny))
eps0_field = ti.field(dtype=ti.f32, shape=(Nx, Ny))

CHy_field = ti.field(dtype=ti.f32, shape=(Nx, Ny))
CHx_field = ti.field(dtype=ti.f32, shape=(Nx, Ny))
CEzHy_field = ti.field(dtype=ti.f32, shape=(Nx, Ny))
CEzHx_field = ti.field(dtype=ti.f32, shape=(Nx, Ny))


# 常数修改
def initialize_constant(relative_mu=0.999992, relative_eps=89.6):
    global c, Dt
    # c = np.sqrt(1 / (mu_0 * eps_0 * relative_mu * relative_eps))
    # Dt = Sc / c * Dx
    print("velocity of light is {cc} and the time step length is {Ts}".format(cc=c, Ts=Dt))


# 空间特征值初始化
@ti.kernel
def initialize_constant_field(relative_mu: ti.f32, relative_eps: ti.f32):
    for i, j in CHy_field:
        mu0_field[i, j] = mu_0 * relative_mu
        eps0_field[i, j] = eps_0 * relative_eps


# 空间电磁学常数特征化
@ti.kernel
def specialize_constant_rectangle(width: ti.int32, height: ti.int32, region_x: ti.int32, region_y: ti.int32):  # 矩形特征
    for i, j in ti.ndrange(width, height):
        ii = i - int(width / 2) + region_x
        jj = j - int(height / 2) + region_y
        mu0_field[ii, jj] = mu_0 * 0.99996
        eps0_field[ii, jj] = eps_0 * 6.9


# 根据特征区实现更新常数
@ti.kernel
def initialize_field_change():
    for i, j in CHx_field:
        CHy_field[i, j] = Dt/mu0_field[i, j]/Dx
        CHx_field[i, j] = Dt/mu0_field[i, j]/Dy
        CEzHx_field[i, j] = Dt/eps0_field[i, j]/Dx
        CEzHy_field[i, j] = Dt/eps0_field[i, j]/Dx


# 更新旧电场值
@ti.func
def refresh_old_electric_field():
    for i, j in Electric_field:
        Electric_field_old[i, j] = Electric_field[i, j]


# 电磁场初始化函数
@ti.kernel
def initialize_all_fields():  # 只在最开始模拟时调用一次
    for i, j in Electric_field:
        Electric_field[i, j] = 0.0
        Magnetic_field_y[i, j] = 0.0
        Magnetic_field_x[i, j] = 0.0
    refresh_old_electric_field()


# 电磁场点状源正弦激励波
@ti.kernel
def electric_pulse_dot_source(t_step_num: ti.int32, source_x: ti.int32, source_y: ti.int32):  # 反复调用
    real_time = t_step_num*Dt
    pulse = ti.sin(omega * real_time)
    Electric_field[source_x, source_y] = Electric_field[source_x, source_y] + pulse * 2.0


# 电磁场杆状源正弦激励波
@ti.kernel
def electric_pulse_rod_source(t_step_num: ti.int32):  # 反复调用
    real_time = t_step_num*Dt
    Source_x = 1200
    pulse = ti.sin(omega * real_time)
    for i in ti.ndrange((0, Ny + 1)):
        Electric_field[Source_x, i] = pulse*0.01


# 运动带电体————简谐运动
@ti.kernel
def electric_simple_harmonic_move_source(v: ti.f32, t_step_num: ti.int32, start_time_step: ti.int32, end_time_step: ti.int32):
    real_time = t_step_num * Dt
    pulse = 0.2
    v = v * ti.cos((t_step_num - start_time_step)/(end_time_step - start_time_step) * 2 * pi)
    if t_step_num < end_time_step and t_step_num >= start_time_step:
        real_movement = real_time * v
        cell_num = int(real_movement / Dx)
        Electric_field[Source_x + cell_num, Source_y] = Electric_field[Source_x + cell_num, Source_y] + pulse  # Electric_field[Source_x + cell_num, Source_y] + pulse
    elif t_step_num >= end_time_step and t_step_num >= start_time_step:
        real_movement = real_time * v
        cell_num = int(real_movement / Dx)
        Electric_field[Source_x + cell_num, Source_y] = Electric_field[Source_x + cell_num, Source_y] + pulse


# 运动带电体————线性运动
@ti.kernel
def electric_linear_move_source(v: ti.f32, t_step_num: ti.int32, start_time_step: ti.int32, end_time_step: ti.int32):
    real_time = t_step_num * Dt
    pulse = 0.2
    if t_step_num < end_time_step and t_step_num >= start_time_step:
        real_movement = real_time * v
        cell_num = int(real_movement / Dx)
        Electric_field[Source_x + cell_num, Source_y] = Electric_field[Source_x + cell_num, Source_y] + pulse
        Electric_field[Source_x + cell_num - 1, Source_y] = Electric_field[Source_x + cell_num, Source_y] - pulse
    elif t_step_num >= end_time_step:
        cell_num = int(end_time_step * Dt * v / Dx)
        Electric_field[Source_x + cell_num, Source_y] = Electric_field[Source_x + cell_num, Source_y] + pulse
        # Electric_field[Source_x + cell_num - 1, Source_y] = Electric_field[Source_x + cell_num, Source_y] - pulse


# 电磁场更新方程
# x方向磁场更新
@ti.func
def update_magnetic_field_x():
    for i in ti.ndrange((0, Nx)):
        for j in ti.ndrange((0, Ny-1)):
            Magnetic_field_x[i, j] = Magnetic_field_x[i, j] - CHx_field[i, j]*(Electric_field[i, j+1] - Electric_field[i, j])


# y方向磁场更新
@ti.func
def update_magnetic_field_y():
    for i in ti.ndrange((0, Nx-1)):
        for j in ti.ndrange((0, Ny)):
            Magnetic_field_y[i, j] = Magnetic_field_y[i, j] + CHy_field[i, j]*(Electric_field[i+1, j] - Electric_field[i, j])


# 电场更新
@ti.func
def update_electric_field():
    for i in ti.ndrange((1, Nx-1)):
        for j in ti.ndrange((1, Ny-1)):
            Electric_field[i, j] = Electric_field[i, j] + CEzHy_field[i, j]*(Magnetic_field_y[i, j] - Magnetic_field_y[i-1, j]) -\
                CEzHx_field[i, j]*(Magnetic_field_x[i, j]-Magnetic_field_x[i, j-1])


# Mur ABC方法 边界值更新
@ti.func
def update_edges():
    for j in ti.ndrange((0, Ny)):
        Electric_field[0, j] = Electric_field_old[1, j] +\
                               (Sc - 1)/(Sc + 1)*(Electric_field[1, j] - Electric_field_old[0, j])
        Electric_field[Nx - 1, j] = Electric_field_old[Nx - 2, j] + \
                                    (Sc - 1) / (Sc + 1) * (Electric_field[Nx - 2, j] - Electric_field_old[Nx - 1, j])
    for i in ti.ndrange((0, Nx)):
        Electric_field[i, Ny - 1] = Electric_field_old[i, Ny - 2] +\
                                  (Sc - 1)/(Sc + 1)*(Electric_field[i, Ny - 2] - Electric_field_old[i, Ny - 1])
        Electric_field[i, 0] = Electric_field_old[i, 1] +\
                               (Sc - 1)/(Sc + 1)*(Electric_field[i, 1] - Electric_field_old[i, 0])
    Electric_field[0, 0] = 0.5*(Electric_field[0, 1] + Electric_field[1, 0])
    Electric_field[Nx-1, 0] = 0.5*(Electric_field[Nx-1, 1] + Electric_field[Nx-2, 0])
    Electric_field[0, Ny-1] = 0.5*(Electric_field[1, Ny-1] + Electric_field[0, Ny-2])
    Electric_field[Nx-1, Ny-1] = 0.5*(Electric_field[Nx-2, Ny-1] + Electric_field[Nx-1, Ny-2])


# 总场更新方程
@ti.kernel
def update_all_fields():  # 反复调用
    update_magnetic_field_x()
    update_magnetic_field_y()
    update_electric_field()
    update_edges()
    refresh_old_electric_field()


# GGUI接口
window = ti.ui.Window("FDTD Simulation on GGUI", (1920, 1080), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1.0, 1.0, 1.0))
scene = ti.ui.Scene()
camera = ti.ui.make_camera()

# GGUI变量
# 三角形顶点
vertices = ti.Vector.field(3, dtype=float, shape=Nx*Ny)
# 三角形边
indices = ti.field(int, shape=(Nx - 1) * (Ny - 1) * 6)
colors = ti.Vector.field(3, dtype=float, shape=Nx*Ny)


# 三角形顶点值获取——来自电场值
@ti.kernel
def update_vertices():  # 反复调用
    for i, j in ti.ndrange(Nx, Ny):
        vertices[i * Nx + j][2] = Electric_field[i, j] * 10
        vertices[i * Nx + j][0] = 0.01 * (int(i - Nx / 2))
        vertices[i * Nx + j][1] = 0.01 * (int(j - Ny / 2))
    for i, j in ti.ndrange(Nx - 1, Ny - 1):
        if vertices[i * Nx + j][2] > 0.05:
            colors[i * Nx + j] = (0.5, 0.25, 0.25)
        elif vertices[i * Nx + j][2] > 0.075:
            colors[i * Nx + j] = (0.5, 0, 0)
        else:
            colors[i * Nx + j] = (0.5, 0.5, 1)


# 三角形三边获取
@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(Nx - 1, Ny - 1):
        quad_id = (i * (Nx - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * Nx + j
        indices[quad_id * 6 + 1] = (i + 1) * Nx + j
        indices[quad_id * 6 + 2] = i * Nx + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * Nx + j + 1
        indices[quad_id * 6 + 4] = i * Nx + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * Nx + j
        # 定义颜色
        colors[i * Nx + j] = (0.5, 0.5, 1)


# 主函数
def main():
    time_step = 0
    initialize_all_fields()
    initialize_constant_field(relative_mu=1.0, relative_eps=1.0)
    specialize_constant_rectangle(width=50, height=100, region_x=150, region_y=500)  # 底边
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=575)  # 上侧
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=425)  # 下侧
    # 生成阵列
    specialize_constant_rectangle(width=50, height=100, region_x=150, region_y=700)
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=775)
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=625)
    specialize_constant_rectangle(width=50, height=100, region_x=150, region_y=300)
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=375)
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=225)
    specialize_constant_rectangle(width=50, height=100, region_x=150, region_y=100)
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=175)
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=25)
    specialize_constant_rectangle(width=50, height=100, region_x=150, region_y=900)
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=975)
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=825)
    specialize_constant_rectangle(width=50, height=100, region_x=150, region_y=1100)
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=1175)
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=1025)
    specialize_constant_rectangle(width=50, height=100, region_x=150, region_y=1300)
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=1375)
    specialize_constant_rectangle(width=300, height=50, region_x=275, region_y=1225)
    initialize_field_change()
    while window.running:
        if time_step == 0:
            initialize_mesh_indices()
        electric_pulse_rod_source(t_step_num=time_step)
        # electric_pulse_dot_source(t_step_num=time_step, source_x=30, source_y=700)
        # electric_linear_move_source(v=2.0e8, t_step_num=time_step, start_time_step=1, end_time_step=800)
        # electric_simple_harmonic_move_source(v=2.0e8, t_step_num=time_step, start_time_step=1, end_time_step=400)
        update_all_fields()
        update_vertices()
        camera.position(0.0, -12.0, 9.0)
        camera.lookat(0.0, 0.0, -2.0)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 6, 12), color=(1, 1, 1))
        scene.mesh(vertices,
                   indices=indices,
                   per_vertex_color=colors,
                   two_sided=True)

        # Draw a smaller ball to avoid visual penetration
        canvas.scene(scene)
        window.show()
        time_step += 1
        if time_step >= Nt:
            time_step = 0


if __name__ == "__main__":
    initialize_constant(relative_mu=0.999992, relative_eps=89.6)
    main()
