from manim import *

class Flowchart(Scene):
    def construct(self):
        # Text elements
        init = MathTex(r"\text{Initialization}\\ \text{One set } \rho = \rho_{\text{beg}}\\ X_0 = X_{\text{init}}").scale(0.5).shift(UP*2)
        increment = MathTex(r"\text{Incrementation}\\ (X = X_0 + \rho)").scale(0.5)
        check_rho = MathTex(r"\rho \leq \rho_{\text{end}} ?").scale(0.5).shift(LEFT*3)
        end = MathTex(r"\text{End}").scale(0.5).shift(LEFT*5)
        compare_f = MathTex(r"\|F(X)\| \leq \|F(X_0)\| ?").scale(0.5).shift(DOWN*2)
        update_rho = MathTex(r"\rho = \frac{\rho}{2}").scale(0.5).shift(LEFT*3 + DOWN*2)
        reset_x = MathTex(r"X=X_0").scale(0.5).shift(RIGHT*2 + DOWN)

        # Surrounding rectangles
        rect_init = SurroundingRectangle(init, buff = .1, color = WHITE)
        rect_increment = SurroundingRectangle(increment, buff = .1, color = WHITE)
        rect_check_rho = SurroundingRectangle(check_rho, buff = .1, color = WHITE)
        rect_end = SurroundingRectangle(end, buff = .1, color = WHITE)
        rect_compare_f = SurroundingRectangle(compare_f, buff = .1, color = WHITE)
        rect_update_rho = SurroundingRectangle(update_rho, buff = .1, color = WHITE)
        rect_reset_x = SurroundingRectangle(reset_x, buff = .1, color = WHITE)

        # Arrows
        arrow_init_to_increment = Arrow(init.get_bottom(), increment.get_top(), buff=0.1)
        arrow_increment_to_check_rho = Arrow(check_rho.get_right(), increment.get_left(), buff=0.1)
        arrow_check_rho_to_update_rho = Arrow(update_rho.get_top(), check_rho.get_bottom(), buff=0.1)
        arrow_check_rho_to_end = Arrow(check_rho.get_left(), end.get_right(), buff=0.1)
        arrow_compare_f_to_reset_x = Arrow(compare_f.get_right(), reset_x.get_bottom(), buff=0.1, path_arc=90*DEGREES)
        arrow_reset_x_to_increment = Arrow(reset_x.get_top(), increment.get_right(), path_arc=90*DEGREES, buff=0.1)
        arrow_increment_to_compare_f = Arrow(increment.get_bottom(), compare_f.get_top(), buff=0.1)
        arrow_compare_f_to_update_rho = Arrow(compare_f.get_left(), update_rho.get_right(), buff=0.1)

        label_check_rho_to_update_rho = Text("No").scale(0.5).next_to(arrow_check_rho_to_update_rho, LEFT, buff=0.1)
        label_check_rho_to_end = Text("Yes").scale(0.5).next_to(arrow_check_rho_to_end, DOWN, buff=0.1)
        label_compare_f_to_reset_x = Text("No").scale(0.5).next_to(arrow_compare_f_to_reset_x, DR*0.1, buff=0.1)
        label_compare_f_to_update_rho = Text("Yes").scale(0.5).next_to(arrow_compare_f_to_update_rho, DOWN, buff=0.1)


        # Adding to scene
        self.add(init, increment, check_rho, end, compare_f, update_rho, reset_x)
        self.add(rect_init, rect_increment, rect_check_rho, rect_end, rect_compare_f, rect_update_rho, rect_reset_x)
        self.add(arrow_init_to_increment, arrow_increment_to_check_rho, arrow_check_rho_to_update_rho, arrow_check_rho_to_end, arrow_compare_f_to_reset_x, arrow_reset_x_to_increment, arrow_increment_to_compare_f, arrow_compare_f_to_update_rho)
        self.add(label_check_rho_to_update_rho, label_check_rho_to_end, label_compare_f_to_reset_x, label_compare_f_to_update_rho)
