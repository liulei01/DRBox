function SelectRotatedTarget(image_name,boxfile)
% SELECTROTATEDTARGET - A tool for selecting rotatable target boxes in images manually
%
%   SELECTROTATEDTARGET(image_name) open image image_name for operating.
%   SELECTROTATEDTARGET(image_name,resolution) open image image_name with specific
%   resolution (default: 1m) for operating.
%   SELECTROTATEDTARGET(image_name,resolution,boxfile) open image image_name with
%   specific resolution (default: 1m) for operating. The result is written
%   into file boxfile (default: image_name.rbox).
%
% Written by LIU Lei, 2017/4/27, revised 2017/5/5, 2017/11/30

if nargin<1
    [filename,pathname]=uigetfile({'*.jpg;*.tif;*.png;*.gif','All Image Files';...
        '*.*','All Files' });
    if filename==0
        return;
    end
    image_name=[pathname,filename];
    [filename,pathname]=uigetfile({'*.txt;*.rbox;*.rbox.score','All rbox Files';...
        '*.*','All Files' });
    boxfile=[pathname,filename];
    if filename==0
        fid = fopen([image_name,'.rbox'], 'wt');
        boxfile = [image_name,'.rbox'];
        fclose(fid);
    end
    
end
resolution=1;

width=25/resolution;
height=14/resolution;
classnum=1;
rotate=0;
stat='N';
mark_mode = 1;
visible=1;
text_visible=0;
figure('WindowKeyPressFcn',@wkpf,'WindowButtonMotionFcn',@wbmcb1,'WindowButtonUpFcn',@wbucb1);
ah=gca;
imshow(image_name);
title('Press H for help');
hold on;
set(ah,'Position',[0 0 1 0.95]);
fid=fopen(boxfile,'r');
box_mat=[];
onshow=[];
point_vec=[];rect_vec=[];text_vec=[];
rec_mode = 0;
box_mat = fscanf(fid,'%f');
if isempty(strfind(boxfile, 'score'))
    box_mat = reshape(box_mat, 6, length(box_mat) / 6)';
else
    box_mat = reshape(box_mat, 7, length(box_mat) / 7)';
    thr = inputdlg('Input a threshold:');
    thr = str2num(thr{1});
    box_mat = box_mat(box_mat(:,7)>thr, :);
    box_mat = box_mat(:,1:6);
end
h = waitbar(0,'Loading...');
for i = 1: size(box_mat, 1)
    onshow=[onshow;1];
    rect_vec=[rect_vec;rRectangle('Position',[box_mat(i,1:4),box_mat(i,6)],'EdgeColor','r')];
    delete(rect_vec(end));
    point_vec=[point_vec;plot(box_mat(i,1),box_mat(i,2),'or')];
    % text_vec=[text_vec;text(box.x,box.y,num2str(box.c),'Color','r','backgroundcolor','g','Visible','off')];
    if mod(i, 10) == 0
        waitbar(i/size(box_mat,1));
    end
end
fclose(fid);
close(h);
selected=[];
selected_old=selected;
rect=[];
mark_mode = 'Select points';
tmpline = [];

    function wkpf(src,callbackdata)
        %global stat; global width; global height
        if strcmp(callbackdata.Key,'space')
            if strcmp(stat,'N')
                title('Press H for help -- Edit mode');
                src.WindowButtonMotionFcn = @wbmcb;
                src.WindowButtonUpFcn = @wbucb;
                src.WindowScrollWheelFcn = @wswf;
                mark_mode = listdlg('ListString', {'Moving rectangle','Select points'},'SelectionMode','single');
                stat='Y';
                if mark_mode == 1
                    cp=ah.CurrentPoint;
                    xinit = cp(1,1);
                    yinit = cp(1,2);
                    pnt=plot(xinit,yinit,'+r');
                    rect=rRectangle('Position',[xinit,yinit,width,height,rotate],'EdgeColor','r');
                    visible=1;
                    ind_onshow = find(onshow);
                    for i=1:length(ind_onshow)
                        if rec_mode
                            set(rect_vec(ind_onshow(i)),'Visible','on');
                        else
                            set(point_vec(ind_onshow(i)),'Visible','on');
                        end
                    end
                end
            end
        elseif strcmp(callbackdata.Character,'p')
            prompt = {'Resolution(m)','Width(m)','Height(m)','Angle','Label'};
            dlg_title = 'Input';
            num_lines = 1;
            def = {num2str(resolution),num2str(width*resolution),num2str(height*resolution),num2str(rotate),num2str(classnum)};
            answer = inputdlg(prompt,dlg_title,num_lines,def);
            if ~isempty(answer)
                resolution=str2num(answer{1});
                width=str2num(answer{2})/resolution;
                height=str2num(answer{3})/resolution;
                classnum=str2num(answer{5});
                rotate=str2num(answer{4});
            end
        elseif strcmp(callbackdata.Character,'w')
            if strcmp(stat,'Y')
                rotate=mod(rotate+1,360);
                wbmcb(src,callbackdata);
            end
        elseif strcmp(callbackdata.Character,'s')
            if strcmp(stat,'Y')
                rotate=mod(rotate-1,360);
                wbmcb(src,callbackdata);
            end
        elseif strcmp(callbackdata.Character,'a')
            if strcmp(stat,'Y')
                rotate=mod(rotate+30,360);
                wbmcb(src,callbackdata);
            end
        elseif strcmp(callbackdata.Character,'d')
            if strcmp(stat,'Y')
                rotate=mod(rotate-30,360);
                wbmcb(src,callbackdata);
            end
        elseif strcmp(callbackdata.Character,'j')
            if strcmp(stat,'Y')
                width=width+1;
                wbmcb(src,callbackdata);
            end
        elseif strcmp(callbackdata.Character,'n')
            if strcmp(stat,'Y')
                width=max(width-1,1);
                wbmcb(src,callbackdata);
            end
        elseif strcmp(callbackdata.Character,'k')
            if strcmp(stat,'Y')
                height=height+1;
                wbmcb(src,callbackdata);
            end
        elseif strcmp(callbackdata.Character,'m')
            if strcmp(stat,'Y')
                height=max(height-1,1);
                wbmcb(src,callbackdata);
            end
        elseif strcmp(callbackdata.Character,'h') || strcmp(callbackdata.Character,'H')
            if strcmp(stat,'N')
                msgbox({'View mode: view and delete boxes',...
                    '  Click left: select a box',...
                    '  Click right: delete the selected box',...
                    '  P: open parameter setting dialog',...
                    '  Z: zoom in',...
                    '  X: zoom out',...
                    '  Direction keys: move image',...
                    '  Space key: switch to edit mode'});
            else
                msgbox({'Edit mode: place boxes',...
                    '  P: open parameter setting dialog',...
                    '  Z: zoom in',...
                    '  X: zoom out',...
                    '  Direction keys: move image',...
                    '  Click right: return to view mode',...
                    '  If you select "Moving rectangle":',...
                    '    WASD/MouseWheel: change angle',...
                    '    JNKM: change size',...
                    '  If you select "Select points":',...
                    '    click the tail and head of the target to determine angle, ',...
                    '    then click top-left and bottom-right corner to determin size.'});
            end
        elseif strcmp(callbackdata.Character,'z')
            position=get(src,'Position');
            ratio=position(3)/position(4)/0.95;%���ڿ?��
            cp=get(ah,'CurrentPoint');
            x=cp(1,1);y=cp(1,2);
            xl=get(ah,'Xlim');yl=get(ah,'Ylim');
            if xl(2)-xl(1)-(yl(2)-yl(1))*ratio>1
                xl=xl*0.9+x*0.1;
                if (xl(2)-xl(1)<ratio*(yl(2)-yl(1)))
                    xl=xl*ratio*(yl(2)-yl(1))/(xl(2)-xl(1));
                end
            elseif xl(2)-xl(1)-(yl(2)-yl(1))*ratio<-1
                yl=yl*0.9+y*0.1;
                if (xl(2)-xl(1)>ratio*(yl(2)-yl(1)))
                    yl=yl*(xl(2)-xl(1))/(ratio*(yl(2)-yl(1)));
                end
            else
                xl=xl*0.9+x*0.1;
                yl=yl*0.9+y*0.1;
            end
            xlim(xl);ylim(yl);
            box_change();
        elseif strcmp(callbackdata.Character,'x')
            cp=get(ah,'CurrentPoint');
            x=cp(1,1);y=cp(1,2);
            xl=get(ah,'Xlim');yl=get(ah,'Ylim');
            xl=xl*1.1-x*0.1;
            yl=yl*1.1-y*0.1;
            xlim(xl);ylim(yl);
            box_change();
        elseif strcmp(callbackdata.Key,'leftarrow')
            xl=get(ah,'Xlim');yl=get(ah,'Ylim');
            xl=xl-(xl(2)-xl(1))*0.1;
            xlim(xl);ylim(yl);
            box_change();
        elseif strcmp(callbackdata.Key,'rightarrow')
            xl=get(ah,'Xlim');yl=get(ah,'Ylim');
            xl=xl+(xl(2)-xl(1))*0.1;
            xlim(xl);ylim(yl);
            box_change();
        elseif strcmp(callbackdata.Key,'uparrow')
            xl=get(ah,'Xlim');yl=get(ah,'Ylim');
            yl=yl-(yl(2)-yl(1))*0.1;
            xlim(xl);ylim(yl);
            box_change();
        elseif strcmp(callbackdata.Key,'downarrow')
            xl=get(ah,'Xlim');yl=get(ah,'Ylim');
            yl=yl+(yl(2)-yl(1))*0.1;
            xlim(xl);ylim(yl);
            box_change();
        end
        
        function box_change()
            onshow1 = box_mat(:,1) >= xl(1) & box_mat(:,1) <= xl(2) & box_mat(:,2) >= yl(1) & box_mat(:,2) <= yl(2);
            if sum(onshow1) < 100 && rec_mode == 0
                rec_mode = 1;
                inds = find(onshow);
                for ii = 1 : length(inds)
                    ind = inds(ii);
                    delete(point_vec(ind));
                end
                onshow = onshow1;
                inds = find(onshow);
                for ii = 1 : length(inds)
                    ind = inds(ii);
                    rect_vec(ind) = rRectangle('Position',[box_mat(ind,1),box_mat(ind,2),box_mat(ind,3),box_mat(ind,4),box_mat(ind,6)],'EdgeColor','r');
                end
                return;
            end
            if sum(onshow1) >= 150 && rec_mode == 1
                rec_mode = 0;
                inds = find(onshow);
                for ii = 1 : length(inds)
                    ind = inds(ii);
                    delete(rect_vec(ind));
                end
                onshow = onshow1;
                inds = find(onshow);
                for ii = 1 : length(inds)
                    ind = inds(ii);
                    point_vec(ind) = plot(box_mat(ind,1),box_mat(ind,2),'or');
                end
                return;
            end
            ind_del = find(onshow - (onshow & onshow1));
            ind_add = find(onshow1 - (onshow & onshow1));
            for ii = 1 : length(ind_del)
                %if isexist(rect_vec(ind_del(ii)))
                if rec_mode
                    delete(rect_vec(ind_del(ii)));
                else
                    delete(point_vec(ind_del(ii)));
                end
                %end
                %if isexist(text_vec(ind_del(ii)))
                %    delete(text_vec(ind_del(ii)));
                %end
            end
            for ii =1 : length(ind_add)
                ind = ind_add(ii);
                if rec_mode
                    rect_vec(ind) = rRectangle('Position',[box_mat(ind,1),box_mat(ind,2),box_mat(ind,3),box_mat(ind,4),box_mat(ind,6)],'EdgeColor','r');
                else
                    point_vec(ind) = plot(box_mat(ind,1),box_mat(ind,2),'or');
                end
                %    text_vec(ind) = text(box_mat(ind,1),box_mat(ind,2),num2str(box_mat(ind,5)),'Color','r','backgroundcolor','g','Visible','off');
            end
            onshow = onshow1;
        end
        
        function wswf(src,callbackdata)
            if mark_mode == 1
                if strcmp(stat,'Y')
                    rotate=mod(rotate+callbackdata.VerticalScrollCount,360);
                    wbmcb(src,callbackdata);
                end
            end
        end
        
        function wbmcb(src,callbackdata)
            if mark_mode == 1
                cp=ah.CurrentPoint;
                xinit = cp(1,1);
                yinit = cp(1,2);
                set(rect,'Position',[xinit,yinit,width,height,rotate]);
                pnt.XData = xinit;
                pnt.YData = yinit;
                %rect=rectangle('Position',[xinit-width/2,yinit-height/2,width,height],'EdgeColor','r');
                drawnow
            elseif mark_mode == 3
                cp=ah.CurrentPoint;
                set(tmpline, 'XData', [xinit, cp(1,1)], 'YData', [yinit, cp(1,2)]);
            elseif mark_mode == 4
                cp = ah.CurrentPoint;
                xd = cp(1,1) - xinit;
                yd = cp(1,2) - yinit;
                tmp = [cosd(rotate), -sind(rotate); sind(rotate), cosd(rotate)]*[xd;yd];
                xd = tmp(1); yd = tmp(2);
                width = 2 * max(xd, 0);
                height = 2 * max(yd, 0);
                set(rect,'Position',[xinit,yinit,width,height,rotate]);
            elseif mark_mode == 5
                cp = ah.CurrentPoint;
                xd = cp(1,1) - xinit;
                yd = cp(1,2) - yinit;
                tmp = [cosd(rotate), -sind(rotate); sind(rotate), cosd(rotate)]*[xd;yd];
                xd = tmp(1); yd = tmp(2);
                width = -min(xd, 0);
                height = -min(yd, 0);
                set(rect,'Position',[(cp(1,1)+xinit)/2,(cp(1,2)+yinit)/2,width,height,rotate]);
            end
        end
        
        function wbucb(src,callbackdata)
            if strcmp(src.SelectionType,'normal')
                if rec_mode == 0
                    return;
                end
                if mark_mode == 1
                    delete(pnt);
                    cp=ah.CurrentPoint;
                    xinit = cp(1,1);
                    yinit = cp(1,2);
                    %                 text1=text(xinit,yinit,num2str(classnum),'Color','r','backgroundcolor','g');
                    %                 if text_visible==1
                    %                     set(text1,'Visible','on');
                    %                 else
                    %                     set(text1,'Visible','off');
                    %                 end
                    pnt=plot(xinit,yinit,'+r');
                    if rec_mode
                        rect_vec=[rect_vec;rect];
                    end
                    rect=rRectangle('Position',[xinit,yinit,width,height,rotate],'EdgeColor','r');
                    box_mat=[box_mat;[xinit,yinit,width,height,classnum,rotate]];
                    onshow=[onshow;1];
                    %text_vec=[text_vec;text1];
                else
                    switch mark_mode
                        case 2
                            cp = ah.CurrentPoint;
                            xinit = cp(1,1);
                            yinit = cp(1,2);
                            tmpline = line([xinit, xinit],[yinit, yinit],'Color','red');
                            mark_mode = 3;
                        case 3
                            cp = ah.CurrentPoint;
                            width = sqrt((cp(1,1)-xinit)^2 + (cp(1,2)-yinit)^2);
                            rotate = -atand((cp(1,2)-yinit)/(cp(1,1)-xinit));
                            if (cp(1,1)-xinit) < 0
                                rotate = rotate + 180;
                            end
                            xinit = (xinit + cp(1,1))/2;
                            yinit = (yinit + cp(1,2))/2;
                            rect = rRectangle('Position',[xinit, yinit, width, height,rotate],'EdgeColor','r');
                            delete(tmpline);
                            mark_mode = 4;
                        case 4
                            cp = ah.CurrentPoint;
                            xinit = cp(1,1);
                            yinit = cp(1,2);
                            mark_mode = 5;
                        case 5
                            cp = ah.CurrentPoint;
                            xd = cp(1,1) - xinit;
                            yd = cp(1,2) - yinit;
                            tmp = [cosd(rotate), -sind(rotate); sind(rotate), cosd(rotate)]*[xd;yd];
                            xd = tmp(1); yd = tmp(2);
                            width = -min(xd, 0);
                            height = -min(yd, 0);
                            xinit = (cp(1,1)+xinit)/2;
                            yinit = (cp(1,2)+yinit)/2;
                            set(rect,'Position',[xinit,yinit,width,height,rotate]);
                            rect_vec=[rect_vec;rect];
                            box_mat = [box_mat;[xinit,yinit,width,height,classnum,rotate]];
                            onshow=[onshow;1];
                            mark_mode = 2;
                    end
                end
            elseif strcmp(src.SelectionType,'alt')
                if mark_mode == 1
                    delete(pnt);
                    delete(rect);
                elseif mark_mode > 2
                    mark_mode = 2;
                    delete(rect);
                end
                stat='N';
                selected=[];selected_old=[];
                title('Press H for help');
                fid=fopen(boxfile,'w');
                if fid>0
                    for i=1:size(box_mat,1)
                        box_vec = box_mat(i,:);
                        box.x=box_vec(1);
                        box.y=box_vec(2);
                        box.w=box_vec(3);
                        box.h=box_vec(4);
                        box.c=box_vec(5);
                        box.r=box_vec(6);
                        fprintf(fid,'%f %f %f %f %d %f\n',box.x,box.y,box.w,box.h,box.c,box.r);
                    end
                    fclose(fid);
                end
                src.WindowButtonMotionFcn = @wbmcb1;
                src.WindowButtonUpFcn = @wbucb1;
                src.WindowScrollWheelFcn = '';
            end
        end
    end

    function wbmcb1(src,callbackdata)
        
        %         cp=ah.CurrentPoint;
        %         x = cp(1,1);
        %         y = cp(1,2);
        %         selected = abs(x-box_mat(:,1))<box_mat(:,3)/2 & abs(y-box_mat(:,2))<box_mat(:,4)/2;
        %         selected = find(selected);
        %         if length(selected)>1
        %             distance = (x-box_mat(:,1)).^2+(y-box_mat(:,2)).^2;
        %             [~,selected] = min(distance);
        %         end
        %         if ~isequal(selected,selected_old) && ~isempty(selected_old)
        %             set(rect_vec(selected_old),'LineWidth',0.5);
        %             drawnow
        %         end
        %         if ~isempty(selected) && isempty(selected_old)
        %             set(rect_vec(selected),'LineWidth',2);
        %             drawnow
        %         elseif ~isempty(selected) && ~isempty(selected_old) && selected ~= selected_old
        %             set(rect_vec(selected),'LineWidth',2);
        %             drawnow
        %         end
        %         selected_old = selected;
    end
    function wbucb1(src,callbackdata)
        if rec_mode == 0
            return;
        end
        if strcmp(src.SelectionType,'normal')
            if isempty(box_mat)
                return;
            end
            cp=ah.CurrentPoint;
            x = cp(1,1);
            y = cp(1,2);
            if isempty(selected)
                selected = abs(x-box_mat(:,1))<box_mat(:,3)/2 & abs(y-box_mat(:,2))<box_mat(:,4)/2;
                selected = find(selected);
                if length(selected)>1
                    distance = (x-box_mat(:,1)).^2+(y-box_mat(:,2)).^2;
                    [~,selected] = min(distance);
                end
                if ~isempty(selected)
                    set(rect_vec(selected),'LineWidth',2);
                    drawnow
                end
            else
                set(rect_vec(selected),'LineWidth',0.5);
                drawnow
                selected=[];
            end
        end
        if strcmp(src.SelectionType,'alt') && ~isempty(selected)
            %delete(text_vec(selected));
            delete(rect_vec(selected));
            %text_vec=[text_vec(1:selected-1);text_vec(selected+1:end)];
            rect_vec=[rect_vec(1:selected-1);rect_vec(selected+1:end)];
            box_mat=[box_mat(1:selected-1,:);box_mat(selected+1:end,:)];
            onshow=[onshow(1:selected-1,:);onshow(selected+1:end,:)];
            selected=[];selected_old=[];
            fid=fopen(boxfile,'w');
            if fid>0
                for i=1:size(box_mat,1)
                    box_vec = box_mat(i,:);
                    box.x=box_vec(1);
                    box.y=box_vec(2);
                    box.w=box_vec(3);
                    box.h=box_vec(4);
                    box.c=box_vec(5);
                    box.r=box_vec(6);
                    fprintf(fid,'%f %f %f %f %d %f\n',box.x,box.y,box.w,box.h,box.c,box.r);
                end
                fclose(fid);
                drawnow
            end
        end
    end
end