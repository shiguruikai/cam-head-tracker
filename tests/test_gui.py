import tkinter as tk
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
    # メインウィンドウ（セッション共有）のイベントで、破棄済みのボタンに再表示しようとしないよう固定を解除する
    if tooltip.is_pinned():
        tooltip.toggle_pin()


def tooltip_windows(button):
    return [w for w in button.winfo_children() if isinstance(w, tk.Toplevel)]


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
