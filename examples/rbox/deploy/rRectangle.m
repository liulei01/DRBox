classdef rRectangle
    properties (SetAccess = protected, GetAccess = private)
        lines
        arrow
    end
    properties
        Position
        LineWidth
        EdgeColor
        Visible
    end
    methods
        function rRect = rRectangle(varargin)
            argin=varargin;
            position=[0 0 1 1 0];
            linewidth=0.5;
            EdgeColor='b';
            while length(argin)>=2
                prop=argin{1};
                val=argin{2};
                argin=argin(3:end);
                switch prop
                    case 'Position', position=val;
                    case 'LineWidth', linewidth=val;
                    case 'EdgeColor', EdgeColor=val;
                    otherwise, continue;
                end
            end
            x=position(1);
            y=position(2);
            w=position(3);
            h=position(4);
            a=position(5);
            rRect.Position=position;
            rRect.EdgeColor=EdgeColor;
            rRect.LineWidth=linewidth;
            coords=repmat([x;y],[1 5])+[cosd(a),sind(a);-sind(a),cosd(a)]*[-w/2 w/2 w/2 -w/2 w/2;-h/2 -h/2 h/2 h/2 0];
            rRect.lines(1)=line([coords(1,1),coords(1,2)],[coords(2,1),coords(2,2)],'Color',EdgeColor,'LineWidth',linewidth);
            rRect.lines(2)=line([coords(1,2),coords(1,3)],[coords(2,2),coords(2,3)],'Color',EdgeColor,'LineWidth',linewidth);
            rRect.lines(3)=line([coords(1,3),coords(1,4)],[coords(2,3),coords(2,4)],'Color',EdgeColor,'LineWidth',linewidth);
            rRect.lines(4)=line([coords(1,1),coords(1,4)],[coords(2,1),coords(2,4)],'Color',EdgeColor,'LineWidth',linewidth);
            rRect.arrow=quiver(x,y,coords(1,5)-x,coords(2,5)-y,'Color',EdgeColor,'LineWidth',0.5,'LineStyle','--','AutoScale','off');
        end
        function delete(rRect)
            for i=1:length(rRect.lines)
                delete(rRect.lines(i));
            end
            delete(rRect.arrow);
        end
        function rRect=set(rRect,varargin)
            argin=varargin;
            while length(argin)>=2
                prop=argin{1};
                val=argin{2};
                argin=argin(3:end);
                switch prop
                    case 'Position'
                        rRect.Position=val;
                        x=val(1);y=val(2);w=val(3);h=val(4);a=val(5);
                        coords=repmat([x;y],[1 5])+[cosd(a),sind(a);-sind(a),cosd(a)]*[-w/2 w/2 w/2 -w/2 w/2;-h/2 -h/2 h/2 h/2 0];
                        set(rRect.lines(1),'XData',[coords(1,1),coords(1,2)],'YData',[coords(2,1),coords(2,2)]);
                        set(rRect.lines(2),'XData',[coords(1,2),coords(1,3)],'YData',[coords(2,2),coords(2,3)]);
                        set(rRect.lines(3),'XData',[coords(1,3),coords(1,4)],'YData',[coords(2,3),coords(2,4)]);
                        set(rRect.lines(4),'XData',[coords(1,1),coords(1,4)],'YData',[coords(2,1),coords(2,4)]);
                        set(rRect.arrow,'XData',x,'YData',y,'UData',coords(1,5)-x,'VData',coords(2,5)-y);
                    case 'LineWidth'
                        rRect.LineWidth=val;
                        for i=1:4
                            set(rRect.lines(i),'LineWidth',val);
                        end
                    case 'EdgeColor'
                        rRect.EdgeColor=val;
                        for i=1:4
                            set(rRect.lines(i),'Color',val);                  
                        end
                        set(rRect.arrow,'Color',val);
                    case 'Visible'
                        rRect.Visible=val;
                        for i=1:4
                            set(rRect.lines(i),'Visible',val);                  
                        end
                        set(rRect.arrow,'Visible',val);
                    otherwise
                        continue;
                end
            end
        end
    end
end