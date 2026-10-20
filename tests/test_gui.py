import ctypes
import sys
import time
import tkinter as tk
from ctypes import wintypes
from tkinter import ttk

import pytest

from cam_head_tracker.gui import HelpTooltip


@pytest.fixture(scope="session")
def root():
    # Windows では Tk インスタンスの生成と破棄を繰り返すと Tcl の初期化に失敗することがあるため、1つだけ生成する
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture(autouse=True)
def callback_errors(root):
    """イベントハンドラ内の例外は Tk が標準エラー出力に出すだけでテストが失敗しないため、集めて検証する。"""
    errors = []
    original = root.report_callback_exception
    root.report_callback_exception = lambda _exc, val, _tb: errors.append(val)
    yield errors
    root.report_callback_exception = original
    assert not errors


@pytest.fixture
def button(root):
    button = ttk.Button(root, text="?")
    button.pack()
    yield button
    # ツールチップのウィンドウも子として一緒に破棄される
    button.destroy()


@pytest.fixture
def tooltip(button):
    tooltip = HelpTooltip(button, "help text")
    yield tooltip
    # メインウィンドウ（セッション共有）のイベントで、破棄済みのウィンドウを操作しないよう固定を解除して閉じる
    if tooltip.is_pinned():
        tooltip.toggle_pin()
    button.event_generate("<Leave>")


@pytest.fixture
def child_widget(root):
    """メインウィンドウ内の別の子ウィジェット（プレビューのキャンバスなど）"""
    child = ttk.Frame(root)
    child.pack()
    yield child
    child.destroy()


@pytest.fixture
def main_window(root):
    """位置や重なり順を検証するための、画面に表示したメインウィンドウとその中のボタン"""
    window = tk.Toplevel(root)
    window.geometry("200x150+100+100")
    button = ttk.Button(window, text="?")
    button.pack()
    window.update()
    yield window, button
    window.destroy()


def tooltip_windows(button):
    return [w for w in button.winfo_children() if isinstance(w, tk.Toplevel)]


def window_handle(window) -> int:
    return int(window.wm_frame(), 16)


def is_window_above(upper, lower) -> bool:
    """Windows の Z オーダーで、upper が lower より手前にあるかを返す。"""
    user32 = ctypes.windll.user32
    user32.GetWindow.argtypes = [wintypes.HWND, ctypes.c_uint]
    user32.GetWindow.restype = wintypes.HWND
    gw_hwndnext = 2
    lower_hwnd = window_handle(lower)
    hwnd = user32.GetWindow(window_handle(upper), gw_hwndnext)
    while hwnd:
        if hwnd == lower_hwnd:
            return True
        hwnd = user32.GetWindow(hwnd, gw_hwndnext)
    return False


def activate_window(window):
    """実際にウィンドウを前面化・アクティブ化し、イベントを処理する。前面化できなかった場合はテストをスキップする。"""
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.restype = wintypes.HWND
    window.lift()
    window.focus_force()
    end = time.perf_counter() + 0.3
    while time.perf_counter() < end:
        window.update()
        time.sleep(0.01)
    if user32.GetForegroundWindow() != window_handle(window):
        # 他のプロセスが前面にある場合など、OS の制限で前面化できないことがある
        pytest.skip("Could not bring the window to the foreground")


def test_tooltip_shows_on_enter_and_hides_on_leave(button, tooltip):
    """マウスを乗せると表示され、離すと閉じることを検証する。"""
    # 実行・検証: マウスを乗せる
    button.event_generate("<Enter>")
    assert tooltip.is_visible()

    # 実行・検証: マウスを離す
    button.event_generate("<Leave>")
    assert not tooltip.is_visible()


def test_tooltip_stays_visible_on_leave_when_pinned(button, tooltip):
    """クリックで固定すると、マウスを離しても表示されたままであることを検証する。"""
    # 準備: マウスを乗せてクリック
    button.event_generate("<Enter>")
    tooltip.toggle_pin()

    # 実行: マウスを離す
    button.event_generate("<Leave>")

    # 検証
    assert tooltip.is_pinned()
    assert tooltip.is_visible()


def test_tooltip_hides_when_unpinned(tooltip):
    """固定中にもう一度クリックすると閉じることを検証する。"""
    # 準備: マウスを乗せずにクリックして固定
    tooltip.toggle_pin()
    assert tooltip.is_visible()

    # 実行: もう一度クリック
    tooltip.toggle_pin()

    # 検証
    assert not tooltip.is_pinned()
    assert not tooltip.is_visible()


def test_tooltip_shows_again_after_unpinned(button, tooltip):
    """固定を解除した後も、マウスを乗せれば再び表示されることを検証する。"""
    # 準備: 固定して解除
    tooltip.toggle_pin()
    tooltip.toggle_pin()

    # 実行
    button.event_generate("<Enter>")

    # 検証
    assert tooltip.is_visible()


@pytest.mark.usefixtures("tooltip")
def test_tooltip_window_is_not_transient(button):
    """
    ツールチップのウィンドウが transient でないことを検証する。
    overrideredirect のウィンドウを transient にすると、Windows では画面に表示されなくなる。
    """
    # 実行
    button.event_generate("<Enter>")

    # 検証
    windows = tooltip_windows(button)
    assert len(windows) == 1
    assert windows[0].wm_overrideredirect()
    assert not windows[0].wm_transient()


def test_tooltip_hides_on_minimize_and_reappears_on_restore_when_pinned(root, tooltip):
    """固定中にメインウィンドウを最小化すると閉じ、復元すると再表示されることを検証する。"""
    # 準備: クリックして固定
    tooltip.toggle_pin()

    # 実行・検証: 最小化
    root.event_generate("<Unmap>")
    assert not tooltip.is_visible()
    assert tooltip.is_pinned()

    # 実行・検証: 復元
    root.event_generate("<Map>")
    assert tooltip.is_visible()


def test_tooltip_stays_hidden_on_restore_when_not_pinned(root, button, tooltip):
    """固定していない場合は、メインウィンドウを復元しても表示されないことを検証する。"""
    # 準備: マウスを乗せてから最小化
    button.event_generate("<Enter>")
    root.event_generate("<Unmap>")

    # 実行
    root.event_generate("<Map>")

    # 検証
    assert not tooltip.is_visible()


def test_pinned_tooltip_ignores_unmap_of_child_widget(child_widget, tooltip):
    """メインウィンドウ内の子ウィジェットが非表示になっても（プレビューの非表示など）、固定表示が閉じないことを検証する。"""
    # 準備
    tooltip.toggle_pin()

    # 実行
    child_widget.event_generate("<Unmap>")

    # 検証
    assert tooltip.is_visible()


def test_tooltip_ignores_map_of_child_widget_while_minimized(root, child_widget, tooltip):
    """最小化中に子ウィジェットが表示されても、ツールチップが再表示されないことを検証する。"""
    # 準備: 固定してから最小化
    tooltip.toggle_pin()
    root.event_generate("<Unmap>")

    # 実行
    child_widget.event_generate("<Map>")

    # 検証
    assert not tooltip.is_visible()


def test_pinned_tooltip_follows_main_window(main_window):
    """固定中にメインウィンドウを移動すると、ツールチップがボタンの左下に追従することを検証する。"""
    # 準備
    window, button = main_window
    tooltip = HelpTooltip(button, "help text")
    tooltip.toggle_pin()
    tip_window = tooltip_windows(button)[0]

    # 実行
    window.geometry("+300+250")
    window.update()

    # 検証
    expected = (button.winfo_rootx(), button.winfo_rooty() + button.winfo_height() + 2)
    assert (tip_window.winfo_rootx(), tip_window.winfo_rooty()) == expected


@pytest.mark.skipif(sys.platform != "win32", reason="Windows のウィンドウの重なり順を検証するため")
def test_pinned_tooltip_stays_in_front_after_returning_from_other_window(root, main_window):
    """
    別のウィンドウに切り替えてからメインウィンドウに戻っても、固定中のツールチップが裏に隠れないことを検証する。
    ツールチップはメインウィンドウに所有されていないため、何もしないとメインウィンドウの裏に隠れる。
    合成イベントではなく実際の前面化で検証する（Windows の Tk では <Activate> が届かないことがあるため）。
    """
    # 準備: 固定表示してから、メインウィンドウとツールチップを覆う別のウィンドウに切り替える
    window, button = main_window
    other = tk.Toplevel(root)
    other.geometry("500x400+50+50")
    try:
        activate_window(window)
        tooltip = HelpTooltip(button, "help text")
        tooltip.toggle_pin()
        tip_window = tooltip_windows(button)[0]
        activate_window(other)

        # 実行: メインウィンドウに戻る
        activate_window(window)

        # 検証
        assert is_window_above(tip_window, window)
    finally:
        other.destroy()
