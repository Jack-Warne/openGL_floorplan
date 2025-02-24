import pygame
import button

class Home():
    def __init__(self):


        screen = pygame.display.set_mode((1600, 900))
        #game variables
        self.game_paused = False

        #define fonts
        font = pygame.font.SysFont("arialblack", 40)

        #define colours
        TEXT_COL = (255, 255, 255)

        #load button images
        resume_img =  pygame.image.load("textures/button/button_resume.png").convert_alpha()
        options_img = pygame.image.load("textures/button/button_options.png").convert_alpha()
        quit_img =    pygame.image.load("textures/button/button_quit.png").convert_alpha()
        video_img =   pygame.image.load('textures/button/button_video.png').convert_alpha()
        audio_img =   pygame.image.load('textures/button/button_audio.png').convert_alpha()
        keys_img =    pygame.image.load('textures/button/button_keys.png').convert_alpha()
        back_img =    pygame.image.load('textures/button/button_back.png').convert_alpha()

        #create button instances
        self.resume_button = button.Button(304, 125, resume_img, 1)
        self.options_button = button.Button(297, 250, options_img, 1)
        self.quit_button = button.Button(336, 375, quit_img, 1)
        self.video_button = button.Button(226, 75, video_img, 1)
        self.audio_button = button.Button(225, 200, audio_img, 1)
        self.keys_button = button.Button(246, 325, keys_img, 1)
        self.back_button = button.Button(332, 450, back_img, 1)

        def draw_text(text, font, text_col, x, y):
            self.img = font.render(text, True, text_col)
            screen.blit(self.img, (x, y))

        def game_loop(self):
            run = True
            while run:

                screen.fill((52, 78, 91))

                #check if game is paused
                if game_paused == True:
                    #check menu state
                    if menu_state == "main":
                        #draw pause screen buttons
                        if self.resume_button.draw(screen):
                            game_paused = False
                        if self.options_button.draw(screen):
                            menu_state = "options"
                        if self.quit_button.draw(screen):
                            run = False
                        #check if the options menu is open
                        if menu_state == "options":
                        #draw the different options buttons
                            if self.video_button.draw(screen):
                                print("Video Settings")
                            if self.audio_button.draw(screen):
                                print("Audio Settings")
                            if self.keys_button.draw(screen):
                                print("Change Key Bindings")
                            if self.__dict__back_button.draw(screen):
                                menu_state = "main"
                else:
                    draw_text("Press SPACE to pause", font, TEXT_COL, 160, 250)

                #event handler
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            game_paused = True
                    if event.type == pygame.QUIT:
                        run = False

            pygame.display.update()

            pygame.quit()