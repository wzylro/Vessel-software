from stl import mesh
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import scrolledtext, simpledialog
import os
import traceback  # 导入错误跟踪模块
import pyvista as pv

pv.set_plot_theme('document')  # 设置主题
from pyvistaqt import BackgroundPlotter  # 替换默认的 Plotter
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
import sys

# 修改 Plotter 的初始化
plotter = BackgroundPlotter()  # 使用 BackgroundPlotter 替代 pv.Plotter()
plotter.close() #初始化完毕关闭窗口

# 全局变量
current_mesh = None
current_mesh_pv = None
plotter = None
highlighted_areas = []
weld_data = {}  # 确保全局变量初始化
highlighting_active = False  # 跟踪高亮状态


def load_model(filepath):
    """加载模型并在导入后高亮焊缝，同时自动弹出窗口"""
    global current_mesh_pv, plotter
    try:
        current_mesh_pv = pv.read(filepath)

        plotter = pv.Plotter()
        plotter.add_mesh(current_mesh_pv, show_edges=True, opacity=0.5)

        for weld_name in weld_data.keys():
            highlight_weld(weld_name)

        plotter.show(auto_close=False)

    except Exception as e:
        print(f"模型加载失败: {str(e)}")


def visualize_stl(file_path):
    global current_mesh, current_mesh_pv, plotter

    try:
        current_mesh = mesh.Mesh.from_file(file_path)

        print(f"正在加载: {os.path.basename(file_path)}")
        print(f"Number of facets: {len(current_mesh)}")
        print(f"Volume: {current_mesh.get_mass_properties()[0]}")
        print(f"Center of mass: {current_mesh.get_mass_properties()[1]}")
        print(f"Moments of inertia: {current_mesh.get_mass_properties()[2]}")

        vertices = current_mesh.vectors.reshape(-1, 3)
        faces = np.hstack(
            [np.array([3, i * 3, i * 3 + 1, i * 3 + 2])
             for i in range(len(current_mesh))]
        ).astype(int)
        current_mesh_pv = pv.PolyData(vertices, faces)

        generate_weld_data()
        update_weld_list()

        if plotter is not None:
            plotter.close()

        plotter = BackgroundPlotter()  # 使用 BackgroundPlotter
        plotter.add_mesh(current_mesh_pv, color='lightblue', show_edges=True,
                         opacity=1.0, name='base_mesh')
        plotter.add_axes()
        plotter.show_grid()

        plotter.enable_cell_picking(callback=on_cell_pick, show_message=False)
        enable_weld_controls()

    except Exception as e:
        status_label.config(text=f"错误: {str(e)}")
        print(f"可视化出错: {str(e)}")
        traceback.print_exc()



def generate_weld_data():
    """生成模拟焊缝数据"""
    global current_mesh_pv, weld_data

    if current_mesh_pv is None:
        return

    n_cells = current_mesh_pv.n_cells
    print(f"模型总单元数: {n_cells}")

    weld_data = {}
    for i in range(12):
        start = int(n_cells * (i / 12))
        end = int(n_cells * ((i + 1) / 12))
        weld_data[f'焊缝{i + 1}'] = {
            'cells': list(range(start, end)),
            'strength': round(np.random.uniform(0.7, 0.95), 2),
            'status': '正常' if np.random.rand() > 0.3 else '需检查',
            'color': np.random.choice(['red', 'green', 'yellow', 'blue', 'orange']),
            'info': f'焊缝 {i + 1} 区域，模拟数据'
        }

    for name, data in weld_data.items():
        print(f"{name} 包含 {len(data['cells'])} 个单元")

    for name, data in weld_data.items():
        valid_cells = [c for c in data['cells'] if c < n_cells]
        weld_data[name]['cells'] = valid_cells
        print(f"焊缝 {name} 包含 {len(valid_cells)} 个有效单元")


def update_weld_list():
    """更新焊缝列表"""
    global weld_list

    weld_list.delete(0, tk.END)

    if 'weld_data' in globals() and weld_data:
        for name, data in weld_data.items():
            status_indicator = "✓" if data['status'] == '正常' else "⚠"
            weld_list.insert(tk.END, f"{status_indicator} {name} ({data['status']})")

    weld_list.config(state=tk.NORMAL)
    print(f"已添加 {len(weld_data)} 个焊缝到列表")


def on_weld_select(event):
    """当用户从列表中选择焊缝时触发"""
    global plotter, highlighted_areas

    selection = weld_list.curselection()
    if not selection:
        return

    selected_index = selection[0]
    selected_item = weld_list.get(selected_index)

    parts = selected_item.split(' ')
    if len(parts) < 2:
        print("列表项格式错误")
        return

    selected_name = parts[1]
    print(f"选择了焊缝: {selected_name}")

    display_weld_info(selected_name)
    highlight_weld(selected_name)


def display_weld_info(weld_name):
    """显示焊缝详细信息，并增加焊缝最大尺寸、焊缝边界和 PCA 结果"""
    if weld_name in weld_data:
        data = weld_data[weld_name]

        info_text.config(state=tk.NORMAL)
        info_text.delete(1.0, tk.END)

        info_text.insert(tk.END, f"名称: {weld_name}\n")
        info_text.insert(tk.END, f"强度: {data['strength']:.2f}\n")
        info_text.insert(tk.END, f"状态: {data['status']}\n")
        info_text.insert(tk.END, f"单元数: {len(data['cells'])}\n")

        if current_mesh_pv is not None and data['cells']:
            try:
                weld_mesh = current_mesh_pv.extract_cells(np.array(data['cells'], dtype=int))
                weld_bounds = weld_mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
                info_text.insert(tk.END, f"焊缝边界: {weld_bounds}\n")
                max_dim = max(weld_bounds[1] - weld_bounds[0],
                              weld_bounds[3] - weld_bounds[2],
                              weld_bounds[5] - weld_bounds[4])
                info_text.insert(tk.END, f"焊缝最大尺寸: {max_dim:.2f}\n")

                points = weld_mesh.points
                mean_point = np.mean(points, axis=0)
                centered = points - mean_point
                cov = np.cov(centered, rowvar=False)
                eig_vals, eig_vecs = np.linalg.eig(cov)
                sorted_indices = np.argsort(eig_vals)[::-1]
                eig_vals = eig_vals[sorted_indices]
                eig_vecs = eig_vecs[:, sorted_indices]
                pca_result = (
                    f"主成分分析 (PCA) 结果:\n"
                    f"  特征值: {np.array2string(eig_vals, precision=2)}\n"
                    f"  主方向:\n    {np.array2string(eig_vecs[:, 0], precision=2)}\n"
                    f"    {np.array2string(eig_vecs[:, 1], precision=2)}\n"
                    f"    {np.array2string(eig_vecs[:, 2], precision=2)}\n"
                )
                info_text.insert(tk.END, pca_result)
            except Exception as e:
                info_text.insert(tk.END, f"计算几何属性出错: {str(e)}\n")

        info_text.insert(tk.END, f"\n信息: {data['info']}\n")
        info_text.insert(tk.END, "\n备注: 此数据为模拟数据，仅供参考。\n")

        if data['status'] == '正常':
            info_text.tag_add("status", "3.0", "3.0 lineend")
            info_text.tag_config("status", foreground="green")
        elif data['status'] == '需检查':
            info_text.tag_add("status", "3.0", "3.0 lineend")
            info_text.tag_config("status", foreground="orange")
        elif data['status'] == '需修复':
            info_text.tag_add("status", "3.0", "3.0 lineend")
            info_text.tag_config("status", foreground="red")

        info_text.config(state=tk.DISABLED)


def highlight_weld(weld_name):
    """高亮或取消高亮显示指定焊缝"""
    global current_mesh_pv, plotter

    if weld_name not in weld_data:
        print(f"错误: {weld_name} 不在焊缝数据中")
        return

    weld_info = weld_data[weld_name]
    if "cells" not in weld_info:
        print(f"错误: {weld_name} 缺少cells字段")
        return

    try:
        actor_name = f"weld_{weld_name}"
        if actor_name in plotter.renderer._actors:
            plotter.remove_actor(actor_name)
            plotter.update()
            plotter.render()
            return

        cells_to_highlight = np.array(weld_info['cells'], dtype=int)
        if cells_to_highlight.size == 0:
            print(f"警告: {weld_name} 无有效单元")
            return

        selected_cells = current_mesh_pv.extract_cells(cells_to_highlight)
        if selected_cells.n_cells == 0:
            print(f"警告: {weld_name} 无有效单元可显示")
            return

        plotter.add_mesh(selected_cells,
                         color=weld_info.get('color'),
                         opacity=1.0,
                         show_edges=True,
                         line_width=4,
                         name=actor_name)

        plotter.update()
        plotter.render()

    except Exception as e:
        print(f"高亮错误: {str(e)}")
        traceback.print_exc()


def on_cell_pick(point):
    global current_mesh_pv

    closest_cell = current_mesh_pv.find_closest_cell(point)
    print(f"点击位置: {point}")
    print(f"最近的单元 ID: {closest_cell}")

    found_weld = None
    for weld_name, weld_info in weld_data.items():
        if closest_cell in weld_info['cells']:
            found_weld = weld_name
            break

    if found_weld:
        print(f"点击的焊缝: {found_weld}")
        highlight_weld(found_weld)


def enable_weld_controls():
    """启用焊缝控制区域"""
    weld_list.config(state=tk.NORMAL)
    info_text.config(state=tk.NORMAL)


def add_modify_delete_buttons():
    """添加修改和删除按钮"""
    global modify_button, delete_button

    button_frame = ttk.Frame(left_frame)
    button_frame.pack(padx=10, pady=5)

    modify_button = ttk.Button(button_frame, text="修改焊缝", command=modify_weld)
    modify_button.pack(side=tk.LEFT, padx=5)

    delete_button = ttk.Button(button_frame, text="删除焊缝", command=delete_weld)
    delete_button.pack(side=tk.LEFT, padx=5)


def modify_weld():
    global weld_data, weld_list, status_label

    selection = weld_list.curselection()
    if not selection:
        status_label.config(text="请先选择要修改的焊缝")
        return

    selected_index = selection[0]
    selected_item = weld_list.get(selected_index)
    selected_name = selected_item.split(' ')[1]

    # 创建修改窗口
    modify_window = tk.Toplevel(root)
    modify_window.title(f"修改焊缝 - {selected_name}")
    modify_window.geometry("600x600")

    # 设置标签样式
    style = ttk.Style()
    style.configure("Custom.TLabel", background="white", foreground="black")

    # 显示原始信息
    data = weld_data[selected_name]

    # 输入框用于强度的修改
    ttk.Label(modify_window, text="强度", style="Custom.TLabel").pack(pady=5)
    strength_entry = ttk.Entry(modify_window)
    strength_entry.pack(pady=5)
    strength_entry.insert(0, str(data['strength']))

    # 输入框用于状态的修改
    ttk.Label(modify_window, text="状态", style="Custom.TLabel").pack(pady=5)
    status_entry = ttk.Entry(modify_window)
    status_entry.pack(pady=5)
    status_entry.insert(0, data['status'])

    # 输入框用于单元数的修改（通常单元数不应手动修改，但这里我们允许修改）
    ttk.Label(modify_window, text="单元数", style="Custom.TLabel").pack(pady=5)
    cells_entry = ttk.Entry(modify_window)
    cells_entry.pack(pady=5)
    cells_entry.insert(0, len(data['cells']))

    # 输入框用于焊缝边界的修改
    ttk.Label(modify_window, text="焊缝边界 (xmin, xmax, ymin, ymax, zmin, zmax)", style="Custom.TLabel").pack(pady=5)
    weld_mesh = current_mesh_pv.extract_cells(np.array(data['cells'], dtype=int))
    weld_bounds = weld_mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
    bounds_entry = ttk.Entry(modify_window)
    bounds_entry.pack(pady=5)
    bounds_entry.insert(0, str(weld_bounds))

    # 输入框用于焊缝最大尺寸的修改
    max_dim = max(weld_bounds[1] - weld_bounds[0],
                  weld_bounds[3] - weld_bounds[2],
                  weld_bounds[5] - weld_bounds[4])
    ttk.Label(modify_window, text="焊缝最大尺寸", style="Custom.TLabel").pack(pady=5)
    max_dim_entry = ttk.Entry(modify_window)
    max_dim_entry.pack(pady=5)
    max_dim_entry.insert(0, f"{max_dim:.2f}")

    # 保存按钮
    save_button = ttk.Button(modify_window, text="保存",
                             command=lambda: save_weld(modify_window, selected_name,
                                                       strength_entry.get(), status_entry.get(),
                                                       cells_entry.get(), bounds_entry.get(),
                                                       max_dim_entry.get()))
    save_button.pack(pady=20)

def save_weld(modify_window, weld_name, strength, status, cells_count, bounds, max_dim):
    global weld_data

    # 更新焊缝信息
    if strength:
        weld_data[weld_name]['strength'] = float(strength)
    if status:
        weld_data[weld_name]['status'] = status
    if cells_count:
        weld_data[weld_name]['cells'] = list(range(int(cells_count)))  # 假设修改为简单的范围
    if bounds:
        # 更新焊缝边界（这里需要解析输入的边界值）
        bounds_values = list(map(float, bounds.strip('()').split(',')))
        weld_data[weld_name]['bounds'] = bounds_values

    # 重新计算最大尺寸（由新的边界决定，可选）
    if len(bounds_values) == 6:
        new_max_dim = max(bounds_values[1] - bounds_values[0],
                          bounds_values[3] - bounds_values[2],
                          bounds_values[5] - bounds_values[4])
        weld_data[weld_name]['max_dim'] = new_max_dim

    update_weld_list()
    modify_window.destroy()
    status_label.config(text=f"焊缝 {weld_name} 信息已更新")


def delete_weld():
    global weld_data, weld_list

    selection = weld_list.curselection()
    if not selection:
        status_label.config(text="请先选择要删除的焊缝")
        return

    selected_index = selection[0]
    selected_item = weld_list.get(selected_index)
    selected_name = selected_item.split(' ')[1]

    confirm = tk.messagebox.askyesno("确认删除", f"确定要删除焊缝 {selected_name} 吗？")
    if confirm:
        del weld_data[selected_name]
        update_weld_list()
        status_label.config(text=f"已删除焊缝 {selected_name}")


def select_stl_file():
    """打开文件选择对话框并加载所选STL文件"""
    file_path = filedialog.askopenfilename(
        title="选择STL文件",
        filetypes=[("STL文件", "*.stl"), ("所有文件", "*.*")],
        initialdir="d:/QQ/Downloads/课程资料"
    )

    if file_path:
        status_label.config(text="正在加载文件...")
        root.after(100, lambda: visualize_stl(file_path))


def extract_corner_welds():
    """提取模型中的角焊缝"""
    global current_mesh_pv, weld_data

    if current_mesh_pv is None:
        status_label.config(text="错误: 请先加载模型")
        return

    try:
        status_label.config(text="正在提取角焊缝...")

        n_cells = current_mesh_pv.n_cells

        corner_weld_data = {}
        for i in range(1, 13):
            start_cell = int(n_cells * (i - 1) / 12)
            end_cell = int(n_cells * i / 12)
            corner_weld_data[f'角焊缝{i}'] = {
                'cells': list(range(start_cell, end_cell)),
                'strength': round(0.6 + 0.05 * (i % 5), 2),
                'status': '正常' if i % 3 != 0 else '需检查',
                'color': 'magenta' if i % 2 == 0 else 'cyan',
                'info': f'自动检测的角焊缝{i}'
            }

        weld_data.update(corner_weld_data)
        update_weld_list()
        status_label.config(text=f"已提取 {len(corner_weld_data)} 个角焊缝")

    except Exception as e:
        status_label.config(text=f"角焊缝提取错误: {str(e)}")
        print(f"角焊缝提取出错: {str(e)}")
        traceback.print_exc()


# 创建主窗口
root = tk.Tk()
root.option_add("*TLabel*foreground", "black")
root.option_add("*TLabel*background", "white")
root.title("船舰智能焊缝检测系统")
root.geometry("1280x720")
root.configure(bg="#2E2E2E")

# 设置全局样式
style = ttk.Style()
style.theme_create("custom", parent="alt", settings={
    "TFrame": {"configure": {"background": "#3A3A3A"}},
    "TLabel": {
        "configure": {
            "background": "#3A3A3A",
            "foreground": "#FFFFFF",
            "font": ("微软雅黑", 12)
        }
    },
    "TButton": {
        "configure": {
            "background": "#4A90E2",
            "foreground": "white",
            "borderwidth": 0,
            "width": 15,
            "font": ("微软雅黑", 11, "bold")
        },
        "map": {
            "background": [("active", "#357ABD"), ("disabled", "#A0A0A0")],
            "foreground": [("disabled", "#E0E0E0")]
        }
    },
    "TListbox": {
        "configure": {
            "background": "#404040",
            "foreground": "#FFFFFF",
            "selectbackground": "#4A90E2",
            "font": ("Consolas", 11)
        }
    },
    "TScrolledText": {
        "configure": {
            "background": "#404040",
            "foreground": "#FFFFFF",
            "insertbackground": "white",
            "font": ("Consolas", 11)
        }
    },
    "TLabelFrame": {
        "configure": {
            "background": "#3A3A3A",
            "foreground": "#4A90E2",
            "relief": "flat",
            "borderwidth": 2,
            "labelmargins": (10, 5),
            "font": ("微软雅黑", 12, "bold")
        }
    },
    "Horizontal.TProgressbar": {
        "configure": {
            "background": "#4A90E2",
            "troughcolor": "#404040",
            "borderwidth": 0,
            "lightcolor": "#6AA8FF",
            "darkcolor": "#357ABD"
        }
    }
})
style.theme_use("custom")

# 创建标题框架
title_frame = ttk.Frame(root)
title_frame.pack(fill=tk.X, pady=15)

# 创建标题标签
title_label = ttk.Label(title_frame,
                        text="⛴️ 船舰智能焊缝检测系统",
                        font=("微软雅黑", 20, "bold"),
                        foreground="#4A90E2")
title_label.pack(expand=True)

# 创建菜单栏
menubar = tk.Menu(root, tearoff=0, bg="#404040", fg="white",
                  activebackground="#4A90E2", activeforeground="white")
root.config(menu=menubar)

# 文件菜单
file_menu = tk.Menu(menubar, tearoff=0, bg="#404040", fg="white")
menubar.add_cascade(label="文件", menu=file_menu)
file_menu.add_command(label="📤 导入模型", command=select_stl_file)
file_menu.add_command(label="🔍 角焊缝提取", command=extract_corner_welds)
file_menu.add_separator()
file_menu.add_command(label="🚪 退出", command=root.quit)

# 主面板布局
main_frame = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

# 左侧控制面板
left_frame = ttk.Frame(main_frame)
main_frame.add(left_frame, weight=1)

# 焊缝列表容器
weld_frame = ttk.LabelFrame(left_frame, text="🔗 焊缝列表")
weld_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

# 自定义滚动条
scroll_style = ttk.Style()
scroll_style.configure("Custom.Vertical.TScrollbar",
                       gripcount=0,
                       background="#404040",
                       troughcolor="#3A3A3A",
                       arrowcolor="white")

# 焊缝列表
weld_list = tk.Listbox(weld_frame,
                       bg="#404040", fg="white",
                       selectbackground="#4A90E2",
                       selectforeground="white",
                       font=("微软雅黑", 11),
                       relief="flat",
                       highlightthickness=0)
scrollbar = ttk.Scrollbar(weld_frame,
                          style="Custom.Vertical.TScrollbar",
                          command=weld_list.yview)
weld_list.config(yscrollcommand=scrollbar.set)
weld_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
weld_list.bind("<<ListboxSelect>>", on_weld_select)

# 添加修改和删除按钮
add_modify_delete_buttons()

# 右侧信息面板
right_frame = ttk.Frame(main_frame)
main_frame.add(right_frame, weight=2)

# 焊缝信息容器
info_frame = ttk.LabelFrame(right_frame, text="📄 焊缝详细信息")
info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

# 信息文本框
info_text = scrolledtext.ScrolledText(info_frame,
                                      wrap=tk.WORD,
                                      bg="#404040",
                                      fg="white",
                                      insertbackground="white",
                                      font=("Consolas", 11),
                                      relief="flat",
                                      padx=10,
                                      pady=10)
info_text.pack(fill=tk.BOTH, expand=True)

# 状态栏
status_bar = ttk.Frame(root, height=25)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

status_label = ttk.Label(status_bar,
                         text="✅ 系统就绪",
                         anchor=tk.W,
                         font=("微软雅黑", 10),
                         foreground="#4A90E2")
status_label.pack(side=tk.LEFT, padx=10)

progress = ttk.Progressbar(status_bar,
                           mode="indeterminate",
                           style="Horizontal.TProgressbar")
progress.pack(side=tk.RIGHT, padx=10, pady=2)


# 动态效果
def animate_title():
    current_color = title_label.cget("foreground")
    new_color = "#6AA8FF" if current_color == "#4A90E2" else "#4A90E2"
    title_label.config(foreground=new_color)
    root.after(1500, animate_title)


animate_title()

# 启动应用
root.mainloop()