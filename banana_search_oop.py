import cv2 as cv
import numpy as np
import math
import pandas as pd
from os import path


class ObjectSearch:

    def __init__(self, label_df: pd.DataFrame, img_features=5000, img_levels=4, seg_features=200, seg_levels=1,
                 detector='orb') -> None:
        """
            Parameters
            ----------

            label_df : pd.DataFrame
                dataframe of labels and bboxes with 'xmin' 'ymin' 'xmax' 'ymax' coords
        """

        self.threshold = None
        self.df_search = None
        self.img_dir = None
        self.fd = None
        self.templates = []
        self.df_ab = []  # [filename, xmin, ymin, xmax, ymax]

        self.label_df = label_df
        self.dir = path.dirname(self.label_df['filename'][0])

        if detector == 'orb' or 'ORB':
            self.im_orb = cv.ORB.create(
                nfeatures=img_features,
                nlevels=img_levels,
                WTA_K=2
            )

            self.seg_orb = cv.ORB.create(
                nfeatures=seg_features,
                nlevels=seg_levels
            )

    @staticmethod
    def _get_img(filename):
        """returns image"""

        im = cv.imread(filename, cv.IMREAD_GRAYSCALE)
        im = cv.Canny(im, 100, 200)

        # cv.imshow('image', im)
        # cv.waitKey(0)

        return im

    def get_templates(self, *rnge, delta=25):
        """gets a set of templates for orb to compare to

            Parameters
            ----------
            *rnge : *args
                args of 3 parameters for range() function
                used to choose template images from csv lable file

            delta : int
                number of pixels to expand bbox in all directions
                usefull if orb can't calculate keypoints

        """

        templates_path = range(*rnge)
        templates_path = [str(x) for x in templates_path]

        for filename in templates_path:

            img = self._get_img(self.dir + '/' + filename + '.png')

            filename = int(filename)

            xmin = self.label_df['xmin'][filename]
            xmax = self.label_df['xmax'][filename]
            ymin = self.label_df['ymin'][filename]
            ymax = self.label_df['ymax'][filename]

            if xmin > delta and ymin > delta:
                img = img[ymin - delta:ymax + delta, xmin - delta:xmax + delta]

                for _ in range(4):
                    img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
                    self.templates.append(img)

        print(f'\n{len(self.templates)} templates were created from {len(self.templates)//4} images')

    def get_search_img(self, *img_num, df_save=None):
        """gets df of images for object detection

            Parameters
            ----------
            *img_num : *args
                args of 3 parameters for range() function

            df_save : str
                filename for dataframe to save csv
        """

        search = []

        for i in range(*img_num):
            filename = self.label_df['filename'][i]

            xmin = self.label_df['xmin'][i]
            ymin = self.label_df['ymin'][i]
            xmax = self.label_df['xmax'][i]
            ymax = self.label_df['ymax'][i]

            line = (
                filename,
                xmin,
                ymin,
                xmax,
                ymax
            )

            search.append(line)

        print(len(search), 'images will be searched in\n')

        self.df_search = pd.DataFrame(search, columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax'])

        if df_save is not None:
            self.df_search.to_csv(df_save)

    def _get_matches(self, im_des, im_kp):
        """calculates matching points for one image for all templates (segments) using orb with threshold on difference distance

        Parameters
        ----------
        im_des : Any
            descriptors for each keypoint of "searched in" images

        im_kp : Sequence[KeyPoint]
            list of KeyPoint objects

        RETURNS
        -------
        dmatch_list : list
            list of indexes of similar keypoints in "searched in" img

        """

        distance_list = []
        kp_list = []

        for seg in self.templates:

            seg_kp, seg_des = self.seg_orb.detectAndCompute(seg, None)

            matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(im_des, seg_des)

            for dmatch in matches:
                if dmatch.distance < self.threshold:
                    distance_list.append(dmatch.distance)
                    kp_list.append(im_kp[dmatch.queryIdx])

        return distance_list, kp_list

    def fit(self, threshold=70, debug=False):
        """calculates and combines all similar keypoints for each img on all templates (segments)

        threshold : int
            keypoints distance threshold

        debug : boolean (Default = True)
            if True then prints number of created keypoints for each img

        """

        self.threshold = threshold

        self.fd = [[], [], []]  # [filename, matched_kp[], distanses[]]

        for filename in self.df_search['filename']:

            im = self._get_img(filename)

            im_kp, im_des = self.im_orb.detectAndCompute(im, None)
            dist_list, kp_list = self._get_matches(
                                                im_des,
                                                im_kp
                                                )

            self.fd[0].append(filename)
            self.fd[1].append(kp_list)
            self.fd[2].append(dist_list)

            if debug:
                print(f'iamge "{filename}" has {len(kp_list)} matching points')

    @staticmethod
    def __get_xy_lists(kp_list):
        """all x and y separeted from keypoint objects"""

        x_list = []
        y_list = []

        for kp in kp_list:

            x, y = kp.pt

            x_list.append(x)
            y_list.append(y)

        return x_list, y_list

    def __get_median_cord(self, kp_list):

        x_list, y_list = self.__get_xy_lists(kp_list)

        try:
            x_mean = round(np.mean(x_list))
            y_mean = round(np.mean(y_list))
        except ValueError:
            x_mean = 0
            y_mean = 0

        return x_mean, y_mean

    @staticmethod
    def __get_abox_img(filename: str, x_mean: int, y_mean: int, dx: int, dy: int, img_save_dir: str) -> tuple:
        """Writes in an image with according anchor box and returns one line of dataframe as a tuple"""

        xmin = x_mean - dx
        ymin = y_mean - dy
        xmax = x_mean + dx
        ymax = y_mean + dy

        im = cv.imread(filename)
        abox = cv.rectangle(im, (ymin, xmin), (ymax, xmax), (0, 0, 255), 2)

        name = img_save_dir + '/' + path.basename(filename)

        abox_obj = (
            name,
            xmin,
            ymin,
            xmax,
            ymax
        )

        cv.imwrite(name, abox)

        return abox_obj

    def get_abox_median(self, abox_side=30, dir_save_im='bananas ds/median', df_save=None) -> pd.DataFrame:
        """calculates and creates anchor box based on median value of all keypoints for each "searched in" img

        Parameters
        ----------

        abox_side : int
            half size of anchor box square

        dir_save_im : str
            directory to the save folder of images with drawn anchor boxes

        df_save : str
            optional filename to save dataframe of created images and their anchor box coordinates (xyxy)


        RETURNS
        -------
        df_ab : pd.DataFrame
            dataframe of created images and their anchor box coordinates (xyxy)
        """

        dx, dy = abox_side, abox_side

        self.df_ab = []  # [filename, xmin, ymin, xmax, ymax]

        for i in range(len(self.fd[0])):

            filename    = self.fd[0][i]
            kp_list     = self.fd[1][i]

            x_mean, y_mean = self.__get_median_cord(kp_list)

            abox_obj = self.__get_abox_img(filename, x_mean, y_mean, dx, dy, dir_save_im)

            self.df_ab.append(abox_obj)

        self.df_ab = pd.DataFrame(self.df_ab, columns=["filename", "xmin", "ymin", "xmax", "ymax"])

        if df_save is not None:
            self.df_ab.to_csv(df_save)

        return self.df_ab

    @staticmethod
    def __dispersion_calc(x_list):

        dsum = 0

        x_mean = np.mean(x_list)

        for x in x_list:
            u = (x - x_mean) ** 2
            dsum += u

        try:
            d = dsum / len(x_list)
            delta = math.sqrt(d)
        except ZeroDivisionError:
            delta = 10
            x_mean = 0

        return delta, x_mean

    def get_aboxes_dispersion(self, dir_save_im='bananas ds/dispersion', df_save=None) -> pd.DataFrame:
        """calculates and creates anchor box based on median value of all keypoints for each "searched in" img

        Parameters
        ----------
        dir_save_im : str
            directory to the save folder of images with drawn anchor boxes

        df_save : str
            optional directory to save dataframe of created images and their anchor box coordinates (xyxy)


        RETURNS
        -------
        df_ab : pd.DataFrame
            dataframe of created images and their anchor box coordinates (xyxy)

        """

        self.df_ab = []  # [filename, xmin, ymin, xmax, ymax]

        for i in range(len(self.fd[0])):

            filename    = self.fd[0][i]
            kp_list     = self.fd[1][i]

            x_list, y_list = self.__get_xy_lists(kp_list)

            dx, x_mean = (np.round(self.__dispersion_calc(x_list))).astype('int')
            dy, y_mean = (np.round(self.__dispersion_calc(y_list))).astype('int')

            abox_obj = self.__get_abox_img(filename, x_mean, y_mean, dx, dy, dir_save_im)

            self.df_ab.append(abox_obj)

        self.df_ab = pd.DataFrame(self.df_ab, columns=["filename", "xmin", "ymin", "xmax", "ymax"])

        if df_save is not None:
            self.df_ab.to_csv(df_save)

        return self.df_ab

    def get_aboxes_non_min_sup(self, abox_side=30, dir_save_im='bananas ds/non_min', df_save=None):
        """calculates and creates anchor box based on minimum value of all keypoints for each "searched in" img

        Parameters
        ----------
        abox_side : int
            half size of anchor box square

        dir_save_im : str
            directory to the save folder of images with drawn anchor boxes

        df_save : str
            optional directory to save dataframe of created images and their anchor box coordinates (xyxy)


        RETURNS
        -------
        df_ab : pd.DataFrame
            dataframe of created images and their anchor box coordinates (xyxy)"""

        self.df_ab = []  # [filename, xmin, ymin, xmax, ymax]

        dx, dy = abox_side, abox_side

        for i in range(len(self.fd[0])):

            filename    = self.fd[0][i]
            kp_list     = self.fd[1][i]
            dist_list   = self.fd[2][i]

            dist_dict = {}  # {kp : dist}

            for kp, dist in zip(kp_list, dist_list):
                dist_dict[kp] = dist

            min_kp = min(dist_dict.keys(), key=dist_dict.get)

            x, y = min_kp.pt

            x = round(x)
            y = round(y)

            abox_obj = self.__get_abox_img(filename, x, y, dx, dy, dir_save_im)

            self.df_ab.append(abox_obj)

        self.df_ab = pd.DataFrame(self.df_ab, columns=["filename", "xmin", "ymin", "xmax", "ymax"])

        if df_save is not None:
            self.df_ab.to_csv(df_save)

        return self.df_ab

    @staticmethod
    def __iou(abox: np.ndarray, bbox: np.ndarray) -> float:
        """Calculates and returns IoU"""

        assert abox.shape[-1] == 4 and bbox.shape[-1] == 4, "Wrong rect size"
        ab = np.stack([abox, bbox]).astype('float32')

        intersect_area = np.maximum(ab[..., 2:].min(axis=0) - ab[..., :2].max(axis=0), 0).prod(axis=-1)
        union_area = ((ab[..., 2] - ab[..., 0]) * (ab[..., 3] - ab[..., 1])).sum(axis=0) - intersect_area

        return intersect_area / union_area

    def get_iou(self) -> float:
        """Calculates Intersection over union """

        pred = np.empty((len(self.df_ab['filename']), 4))
        real = np.empty((len(self.df_ab['filename']), 4))

        for i in range(len(self.df_ab['filename'])):
            pred[i][0] = self.df_ab['xmin'][i]
            pred[i][1] = self.df_ab['ymin'][i]
            pred[i][2] = self.df_ab['xmax'][i]
            pred[i][3] = self.df_ab['ymax'][i]

            real[i][0] = self.df_search['xmin'][i]
            real[i][1] = self.df_search['ymin'][i]
            real[i][2] = self.df_search['xmax'][i]
            real[i][3] = self.df_search['ymax'][i]

        return self.__iou(pred, real)


def fix_csv(csv_filename: str, img_dir: str, csv_save_filename=None) -> pd.DataFrame:
    """concatenates inputed image directory and filename"""

    df = pd.read_csv(csv_filename)
    df = df.rename(columns={"img_name": "filename"})
    df['filename'] = img_dir + '/' + df['filename']

    if csv_save_filename is not None:
        df.to_csv(csv_save_filename)

    return df


def main():
    # fix_csv(csv_filename="bananas ds/label.csv", img_dir='bananas ds/images', csv_save_filename='bananas ds/img_n_bbox.csv')
    csv_df = pd.read_csv('bananas ds/img_n_bbox.csv', index_col=0)

    qw = ObjectSearch(csv_df)

    qw.get_templates(10, 99, 2)
    qw.get_search_img(100, 999, 20, df_save="bananas ds/searched_in_img.csv")

    qw.fit(threshold=60, debug=False)

    qw.get_abox_median(dir_save_im='bananas ds/median', abox_side=35, df_save="bananas ds/csv/abox_median.csv")
    iou = qw.get_iou()
    print('Intersection over union for median method:', np.mean(iou))

    qw.get_aboxes_dispersion(dir_save_im='bananas ds/dispersion', df_save="bananas ds/csv/aboxes_dispersion.csv")
    iou = qw.get_iou()
    print('Intersection over union for dispersion method:', np.mean(iou))

    qw.get_aboxes_non_min_sup(df_save='bananas ds/csv/aboxes_non_min_sup.csv')                                             # save_csv_filename='D:/Учебная/2-МГ-6/Python/Object_Detection/bananas ds/non_min_abox.csv'
    iou = qw.get_iou()
    print('Intersection over union for non min supression method:', np.mean(iou))


if __name__ == '__main__':
    main()
