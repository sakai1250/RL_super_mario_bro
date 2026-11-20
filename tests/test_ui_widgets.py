from PySide6.QtTest import QTest

from mario_app.ui.style import PALETTE, load_pixel_font
from mario_app.ui.widgets import BoxPanel, Key, Lcd, Led


def test_load_pixel_font_registers_dotgothic(qapp):
    assert load_pixel_font() == "DotGothic16"


def test_palette_has_spec_colors():
    assert PALETTE["body"] == "#E6DEC6" and PALETTE["maroon"] == "#5C1B14"
    assert PALETTE["lcd"] == "#B9C29B" and PALETTE["led_on"] == "#E5532D"


def test_lcd_width_is_stable_when_text_changes(qapp):
    lcd = Lcd("0", chars=8)
    w = lcd.sizeHint().width()
    lcd.set_text("12345678")
    assert lcd.sizeHint().width() == w and lcd.text() == "12345678"


def test_key_toggles_and_ignores_clicks_when_disabled(qapp):
    k = Key("学習", checkable=True)
    k.click()
    assert k.isChecked()
    k.setEnabled(False)
    k.click()
    assert k.isChecked()


def test_led_flash_turns_off_after_the_interval(qapp):
    led = Led("保存")
    led.flash(40)
    assert led.is_on()
    QTest.qWait(150)
    assert not led.is_on()


def test_box_panel_renders(qapp):
    box = BoxPanel("SPEED／速度")
    box.resize(300, 100)
    pix = box.grab()
    assert not pix.isNull() and pix.width() == 300
