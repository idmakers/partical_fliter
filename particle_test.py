import pdb
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_mse as mse


# def ssim(src,dst,multichannel=False):
#     print mse(src, dst)


IMG_PATH = ".\{index}.jpg"
FRAME_COUNT = 680

NUM_PARTICLES = 150
NUM_OF_STATE = 4
IDBBX = 1363
IDBBY = 569

WINDOW_WIDTH =104

WINDOW_HEIGHT = 241

S_X_COOR = 0
S_Y_COOR = 1
S_X_VELOCITY = 2
S_Y_VELOCITY = 3
img_array = []
img_EPOCH =[]

for x_V in range(-1,2):
    #for y_V in range(-1,2):
        y_V = 1
        def add_noise(mat):
            return mat + np.random.normal(0, 0.2, mat.shape[0]*mat.shape[1]).reshape(mat.shape)

        def cut_sub_portion(mat, state):
            x_start_index = int(state[S_X_COOR]) - WINDOW_WIDTH//2
            x_end_index = int(state[S_X_COOR]) + WINDOW_WIDTH//2

            y_start_index = int(state[S_Y_COOR]) - WINDOW_HEIGHT//2
            y_end_index = int(state[S_Y_COOR]) + WINDOW_HEIGHT//2
            
            if y_start_index<0:
                    y_start_index = -1*y_start_index
                    y_end_index =y_end_index+(2*y_start_index)
            if y_end_index >mat.shape[0]:
                y_start_index = y_start_index-(y_end_index-mat.shape[0])
                y_end_index =mat.shape[0]
            if x_start_index<0:
                    x_start_index = -1*x_start_index
                    x_end_index =x_end_index+(2*x_start_index)
            if x_end_index >mat.shape[1]:
                x_start_index = x_start_index-(x_end_index-mat.shape[1])
                x_end_index =mat.shape[1]


    
            #print ("X:%d:%d Y:%d:%d SIZE:%f"%(x_start_index, x_end_index,y_start_index,y_end_index,img.shape))

            return img[y_start_index:y_end_index,x_start_index:x_end_index],(x_start_index, y_start_index),(x_end_index, y_end_index)

        init_x  = IDBBX+WINDOW_WIDTH//2
        init_y =  IDBBY+WINDOW_HEIGHT//2
        s_init = [init_x,init_y,x_V,y_V]  # todo change this
        img_frame =0
        S = np.transpose(np.ones((NUM_PARTICLES,NUM_OF_STATE)) * s_init)
        # S = add_noise(S)

        for i_FRAME in range(1,FRAME_COUNT):
            # load first image
            if i_FRAME < 10 :
                filename="E:/OBJECT_DECTECT/MOT16/train/MOT16-04/img1/00000"+str(i_FRAME)+".jpg"
            elif i_FRAME<100:
                filename="E:/OBJECT_DECTECT/MOT16/train/MOT16-04/img1/0000"+str(i_FRAME)+".jpg"
            elif i_FRAME<1000:
                filename="E:/OBJECT_DECTECT/MOT16/train/MOT16-04/img1/000"+str(i_FRAME)+".jpg"
            else:
                filename="E:/OBJECT_DECTECT/MOT16/train/MOT16-04/img1/00"+str(i_FRAME)+".jpg"
            
            print(i_FRAME,end='\n')  
            img = cv2.imread(filename)
            img_sub,(x_start_index, y_start_index),(x_end_index, y_end_index) = cut_sub_portion(img, s_init)
            if i_FRAME == 0:
                org_sub = img_sub
            w = []  # weighs / distances
            # print img_sub.shape
            for i_PARTICLE in range(NUM_PARTICLES):
                noisy_sub, (x_s, y_s), (x_e, y_e) = cut_sub_portion(img, S[:,i_PARTICLE])
                cv2.rectangle(img, (x_s, y_s), (x_e, y_e), (0,255,0), 2)
                # w.append(ssim(org_sub, noisy_sub, multichannel=True)) # org <=> img
                w.append(ssim(img_sub, noisy_sub, multichannel=True)) # org <=> img

            w = np.array(w)
            w = np.square(w)
            w = w / np.sum(w)  # normalize
            c = np.cumsum(w)  # cumulative weights

            # choose and initialize the next step
            # i_max = np.argmax(w)
            # s_init = S[:,i_max]  # add S_init velocity
            s_init = np.dot(S, w)
            s_init[S_X_COOR:S_Y_COOR+1] += s_init[S_X_VELOCITY:S_Y_VELOCITY+1]  # add the velocity


            # choose the particle points for the next step
            # the chance to pick a point is proportional to it's weight - since we pick from cumulative weights according to uniform distribution
            S_next = np.zeros(S.shape)
            for i_PARTICLE in range(NUM_PARTICLES):
                uniform_random = np.random.uniform()
                min_val = min(val for val in c if val > uniform_random)
                j = np.where(c == min_val)[0][0]
                S_next[:,i_PARTICLE] = S[:,j]

            # add the velocity
            S_next = add_noise(S_next)
            S_next[S_X_COOR:S_Y_COOR+1,:] += S_next[S_X_VELOCITY:S_Y_VELOCITY+1,:]  # S_next[0:2,:] += S_next[2:4,:]

            # add noise to take area around repeating points
            S = S_next
            height, width, layers = img.shape
            size = (width,height)
            cv2.rectangle(img, (x_start_index, y_start_index), (x_end_index, y_end_index), (0,0,255), 2)
            img_array.append(img)
            img_EPOCH.append(img)
            img_frame +=1
            if img_frame == 100:    
                out = cv2.VideoWriter("TEST_GT01_x"+str(x_V)+"y"+str(y_V)+"EPOCH"+str(i_FRAME)+'.mp4',cv2.VideoWriter_fourcc(*'avc1'), 15, size)
                for i in range(len(img_EPOCH)):
                    out.write(img_EPOCH[i])
                out.release()
                img_frame=0
                img_EPOCH.clear()

                
        out = cv2.VideoWriter("TEST_GT01_x"+str(x_V)+"y"+str(y_V)+'.mp4',cv2.VideoWriter_fourcc(*'avc1'), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        img_EPOCH.clear()
        img_array.clear()



