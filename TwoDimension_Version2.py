# 使用taichi模拟电磁波在二维空间中的传播
# 导入包
import taichi as ti
# taichi激活
ti.init(arch=ti.vulkan)
# 常数设置区
# 当前只考虑均匀空间
c = 3e8  # 真空光速
pi = 3.1415926  # 圆周率
mu_0 = 4.0*pi*1.0e-7  # 真空磁导率
eps_0 = 8.854e-12  # 真空介电常数
Nx = 1000  # x轴向上的Yee元胞数量
Ny = 1000  # y轴向上的Yee元胞数量
Nt = 3000  # 最长时间步数
f_0 = 5e9  # 激励源频率
omega = 2.0*pi*f_0  # 激励源角频率
Dx = (c/f_0)/2  # Yee元胞在x方向上的长度
Dy = (c/f_0)/2  # Yee元胞在y方向上的长度
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


# 电磁场正弦激励波
@ti.kernel
def electric_pulse_source(t_step_num: ti.int32):  # 反复调用
    real_time = t_step_num*Dt
    cycles = 1
    pulse = 0
    # if real_time < cycles*2.0*pi/omega:
    pulse = 1.0*ti.cos(omega * real_time / 200)
    # Electric_field[Source_x, Source_y] = Electric_field[Source_x, Source_y] + pulse
    for i in ti.ndrange(20, 10):
        Electric_field[Source_x + i - 100, Source_y] = Electric_field[Source_x + i - 100, Source_y]\
                                                                 + pulse*1.0


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
            Magnetic_field_x[i, j] = Magnetic_field_x[i, j] - CHx*(Electric_field[i, j+1] - Electric_field[i, j])


# y方向磁场更新
@ti.func
def update_magnetic_field_y():
    for i in ti.ndrange((0, Nx-1)):
        for j in ti.ndrange((0, Ny)):
            Magnetic_field_y[i, j] = Magnetic_field_y[i, j] + CHy*(Electric_field[i+1, j] - Electric_field[i, j])


# 电场更新
@ti.func
def update_electric_field():
    for i in ti.ndrange((1, Nx-1)):
        for j in ti.ndrange((1, Ny-1)):
            Electric_field[i, j] = Electric_field[i, j] + CEzHy*(Magnetic_field_y[i, j] - Magnetic_field_y[i-1, j]) -\
                CEzHx*(Magnetic_field_x[i, j]-Magnetic_field_x[i, j-1])


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
window = ti.ui.Window("FDTD Simulation on GGUI", (1024, 1024), vsync=True)
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
        vertices[i * Nx + j][0] = Dx * (int(i - Nx / 2))
        vertices[i * Nx + j][1] = Dx * (int(j - Ny / 2))
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
    while window.running:
        if time_step == 0:
            initialize_mesh_indices()
        electric_pulse_source(t_step_num=time_step)
        # electric_linear_move_source(v=2.0e8, t_step_num=time_step, start_time_step=1, end_time_step=800)
        # electric_simple_harmonic_move_source(v=2.0e8, t_step_num=time_step, start_time_step=1, end_time_step=400)
        update_all_fields()
        update_vertices()
        camera.position(0.0/f_0*10e9, -20.0/f_0*10e9, 15.0/f_0*10e9)
        camera.lookat(0.0, 0.0, -2.0)
        scene.set_camera(camera)

        scene.point_light(pos=(0, 10/f_0*10e9, 20/f_0*10e9), color=(1, 1, 1))
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
    main()
